/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.codecs.isoquant;

import static org.apache.lucene.codecs.isoquant.IsoQuantFlatFormat.VECTOR_DATA_EXTENSION;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.packed.DirectMonotonicReader;

public final class IsoQuantFlatReader extends FlatVectorsReader {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(IsoQuantFlatReader.class);

  private final Map<String, FieldEntry> fields = new HashMap<>();
  private final IndexInput vectorData;
  private final IndexInput rawData;
  private final IsoQuantFlatVectorScorer vectorsScorer;
  private final long seed;

  public IsoQuantFlatReader(SegmentReadState state, FlatVectorsScorer scorer, long seed)
      throws IOException {
    super(scorer);
    this.vectorsScorer = (IsoQuantFlatVectorScorer) scorer;
    this.seed = seed;
    int versionMeta = -1;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, IsoQuantFlatFormat.META_EXTENSION);
    boolean success = false;
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
      versionMeta =
          CodecUtil.checkIndexHeader(
              meta,
              IsoQuantFlatFormat.META_CODEC_NAME,
              IsoQuantFlatFormat.VERSION_START,
              IsoQuantFlatFormat.VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      readFields(meta, state.fieldInfos);
      CodecUtil.checkFooter(meta);
      success = true;
    } catch (Throwable t) {
      throw t;
    }

    String dataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, VECTOR_DATA_EXTENSION);
    this.vectorData = state.directory.openInput(dataFileName, state.context);

    IndexInput rawDataInput = null;
    try {
      String rawDataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, IsoQuantFlatFormat.RAW_DATA_EXTENSION);
      rawDataInput = state.directory.openInput(rawDataFileName, state.context);
    } catch (IOException e) {
      // Ignore if not present
    }
    this.rawData = rawDataInput;

    success = false;
    try {
      int versionVectorData =
          CodecUtil.checkIndexHeader(
              vectorData,
              IsoQuantFlatFormat.VECTOR_DATA_CODEC_NAME,
              IsoQuantFlatFormat.VERSION_START,
              IsoQuantFlatFormat.VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      if (versionMeta != versionVectorData) {
        throw new CorruptIndexException(
            "Format versions mismatch: meta=" + versionMeta + ", data=" + versionVectorData,
            vectorData);
      }
      if (rawData != null) {
        CodecUtil.checkIndexHeader(
            rawData,
            IsoQuantFlatFormat.RAW_DATA_CODEC_NAME,
            IsoQuantFlatFormat.VERSION_START,
            IsoQuantFlatFormat.VERSION_CURRENT,
            state.segmentInfo.getId(),
            state.segmentSuffix);
      }
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileSuppressingExceptions((Throwable) null, vectorData, rawData);
      }
    }
  }

  private void readFields(ChecksumIndexInput meta, FieldInfos infos) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = infos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      FieldEntry fieldEntry = readField(meta);
      validateFieldEntry(info, fieldEntry);
      fields.put(info.name, fieldEntry);
    }
  }

  private void validateFieldEntry(FieldInfo info, FieldEntry fieldEntry) {
    int dimension = info.getVectorDimension();
    if (dimension != fieldEntry.dimension) {
      throw new IllegalStateException("Inconsistent vector dimension");
    }
    long vectorBytes = (dimension * fieldEntry.encoding.bits / 8) + 8 + (fieldEntry.compress == IsoCompress.NONE ? ((long) dimension * Float.BYTES) : 0L);
    long numBytes = (long) fieldEntry.size() * vectorBytes;
    if (numBytes != fieldEntry.vectorDataLength) {
      throw new IllegalStateException("Vector data length mismatch");
    }
  }

  private FieldEntry readField(IndexInput input) throws IOException {
    VectorEncoding vectorEncoding = VectorEncoding.values()[input.readInt()];
    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.values()[input.readInt()];
    IsoEncoding encoding = IsoEncoding.fromWireNumber(input.readVInt());
    IsoVariant variant = IsoVariant.fromWireNumber(input.readVInt());
    IsoCompress compress = IsoCompress.fromWireNumber(input.readVInt());
    long vectorDataOffset = input.readVLong();
    long vectorDataLength = input.readVLong();
    long rawDataOffset = 0L;
    long rawDataLength = 0L;
    if (compress != IsoCompress.NONE) {
      rawDataOffset = input.readVLong();
      rawDataLength = input.readVLong();
    }
    int dimension = input.readVInt();
    int size = input.readInt();
    long rawOffsetsDataOffset = 0L;
    long rawOffsetsDataLength = 0L;
    DirectMonotonicReader.Meta rawOffsetsMeta = null;
    if (compress != IsoCompress.NONE) {
      int numBlocks = input.readVInt();
      rawOffsetsDataOffset = input.readLong();
      rawOffsetsMeta = DirectMonotonicReader.loadMeta(input, numBlocks, IsoQuantFlatFormat.DIRECT_MONOTONIC_BLOCK_SHIFT);
      rawOffsetsDataLength = input.readLong();
    }
    OrdToDocDISIReaderConfiguration configuration =
        OrdToDocDISIReaderConfiguration.fromStoredMeta(input, size);
    return new FieldEntry(
        dimension,
        size,
        vectorEncoding,
        similarityFunction,
        encoding,
        variant,
        compress,
        vectorDataOffset,
        vectorDataLength,
        rawDataOffset,
        rawDataLength,
        rawOffsetsDataOffset,
        rawOffsetsDataLength,
        rawOffsetsMeta,
        configuration);
  }

  @Override
  public void checkIntegrity() throws IOException {
    CodecUtil.checksumEntireFile(vectorData);
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    FieldEntry fieldEntry = fields.get(field);
    if (fieldEntry == null || fieldEntry.vectorEncoding != VectorEncoding.FLOAT32) {
      return null;
    }
    IndexInput slice = vectorData.slice("vector_data", fieldEntry.vectorDataOffset, fieldEntry.vectorDataLength);
    DirectMonotonicReader rawOffsetsReader = null;
    if (fieldEntry.rawOffsetsMeta != null) {
      IndexInput offsetsSlice = vectorData.slice("raw_offsets", fieldEntry.rawOffsetsDataOffset, fieldEntry.rawOffsetsDataLength);
      rawOffsetsReader = DirectMonotonicReader.getInstance(fieldEntry.rawOffsetsMeta, offsetsSlice.randomAccessSlice(0, offsetsSlice.length()));
    }
    return IsoQuantOffHeapVectorValues.load(
        fieldEntry.similarityFunction,
        vectorsScorer,
        fieldEntry.ordToDoc,
        fieldEntry.vectorEncoding,
        fieldEntry.dimension,
        fieldEntry.vectorDataOffset,
        fieldEntry.vectorDataLength,
        vectorData,
        rawData == null ? null : rawData.slice("raw_data", fieldEntry.rawDataOffset, fieldEntry.rawDataLength),
        rawOffsetsReader,
        fieldEntry.encoding,
        fieldEntry.variant,
        fieldEntry.compress,
        seed,
        fieldEntry.size);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    return null; // Not supported in IsoQuant
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
    FieldEntry fieldEntry = fields.get(field);
    if (fieldEntry == null) return null;
    return vectorsScorer.getRandomVectorScorer(
        fieldEntry.similarityFunction, getFloatVectorValues(field), target);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
    throw new UnsupportedOperationException("IsoQuant does not support byte targets");
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(vectorData, rawData);
  }

  @Override
  public long ramBytesUsed() {
    return SHALLOW_SIZE
        + RamUsageEstimator.sizeOfMap(
            fields, RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class));
  }

  public IsoQuantFlatVectorScorer getScorer() {
    return vectorsScorer;
  }

  private record FieldEntry(
      int dimension,
      int size,
      VectorEncoding vectorEncoding,
      VectorSimilarityFunction similarityFunction,
      IsoEncoding encoding,
      IsoVariant variant,
      IsoCompress compress,
      long vectorDataOffset,
      long vectorDataLength,
      long rawDataOffset,
      long rawDataLength,
      long rawOffsetsDataOffset,
      long rawOffsetsDataLength,
      DirectMonotonicReader.Meta rawOffsetsMeta,
      OrdToDocDISIReaderConfiguration ordToDoc) {}
}
