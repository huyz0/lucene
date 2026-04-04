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

import static org.apache.lucene.codecs.isoquant.IsoQuantFlatFormat.DIRECT_MONOTONIC_BLOCK_SHIFT;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.compress.LZ4;
import org.apache.lucene.util.packed.DirectMonotonicWriter;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.RotorQuant;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

public final class IsoQuantFlatWriter extends FlatVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      RamUsageEstimator.shallowSizeOfInstance(IsoQuantFlatWriter.class);

  private final SegmentWriteState segmentWriteState;
  private final IndexOutput meta, vectorData, rawData;

  private final List<FieldWriter> fields = new ArrayList<>();

  private static class IsoQuantMergeSub extends org.apache.lucene.index.DocIDMerger.Sub {
    final FloatVectorValues values;
    final KnnVectorValues.DocIndexIterator iterator;
    final boolean fastPath;

    IsoQuantMergeSub(MergeState.DocMap docMap, FloatVectorValues values, boolean fastPath) {
      super(docMap);
      this.values = values;
      this.iterator = values.iterator();
      this.fastPath = fastPath;
    }

    @Override
    public int nextDoc() throws IOException {
      return iterator.nextDoc();
    }

    public int index() {
      return iterator.index();
    }
  }
  private boolean finished;
  private final IsoEncoding encoding;
  private final IsoVariant variant;
  private final IsoCompress compress;
  private final long seed;

  public IsoQuantFlatWriter(SegmentWriteState state, FlatVectorsScorer scorer, IsoEncoding encoding, IsoVariant variant, IsoCompress compress, long seed)
      throws IOException {
    super(scorer);
    this.encoding = encoding;
    this.variant = variant;
    this.compress = compress;
    if (compress == IsoCompress.ZSTD || compress == IsoCompress.LZAV) {
      throw new UnsupportedOperationException(compress + " requires external SPI implementation and is not natively supported in lucene/core");
    }
    this.seed = seed;
    segmentWriteState = state;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, IsoQuantFlatFormat.META_EXTENSION);

    String vectorDataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, IsoQuantFlatFormat.VECTOR_DATA_EXTENSION);

    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      vectorData = state.directory.createOutput(vectorDataFileName, state.context);
      if (compress != IsoCompress.NONE) {
        String rawDataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, IsoQuantFlatFormat.RAW_DATA_EXTENSION);
        rawData = state.directory.createOutput(rawDataFileName, state.context);
      } else {
        rawData = null;
      }

      CodecUtil.writeIndexHeader(
          meta,
          IsoQuantFlatFormat.META_CODEC_NAME,
          IsoQuantFlatFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          vectorData,
          IsoQuantFlatFormat.VECTOR_DATA_CODEC_NAME,
          IsoQuantFlatFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      if (rawData != null) {
        CodecUtil.writeIndexHeader(rawData, IsoQuantFlatFormat.RAW_DATA_CODEC_NAME, IsoQuantFlatFormat.VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
      }
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, this);
      throw t;
    }
  }

  @Override
  public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    if (fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32) {
      throw new IllegalArgumentException("IsoQuant only supports FLOAT32 vectors");
    }
    FieldWriter newField = new FieldWriter(fieldInfo);
    fields.add(newField);
    return newField;
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    for (FieldWriter field : fields) {
      if (sortMap == null) {
        writeField(field, maxDoc);
      } else {
        writeSortingField(field, maxDoc, sortMap);
      }
      field.finish();
    }
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    if (meta != null) {
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (vectorData != null) {
      CodecUtil.writeFooter(vectorData);
    }
    if (rawData != null) {
      CodecUtil.writeFooter(rawData);
    }
  }

  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    for (FieldWriter field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }

  private void writeField(FieldWriter fieldData, int maxDoc) throws IOException {
    long vectorDataOffset = vectorData.alignFilePointer(64);
    long rawDataOffset = rawData == null ? 0 : rawData.alignFilePointer(64);

    int dim = fieldData.fieldInfo.getVectorDimension();
    float[] codebook = RotorQuant.generateCodebook(seed, dim, variant);
    float[] centroids = RotorQuant.getLloydMaxCentroids(encoding.bits, dim);

    int count = fieldData.docsWithField.cardinality();
    int blockSize = 1 << IsoQuantFlatFormat.LZ4_BLOCK_SHIFT;
    int numBlocks = (count + blockSize - 1) / blockSize;
    long[] rawOffsets = compress != IsoCompress.NONE ? new long[numBlocks] : null;
    int blockOrd = 0;
    
    float[] blockBuffer = compress != IsoCompress.NONE ? new float[blockSize * dim] : null;
    int vectorsInBlock = 0;

    for (float[] v : fieldData.vectors) {
      writePackedVector(v, codebook, centroids);
      if (compress != IsoCompress.NONE) {
        System.arraycopy(v, 0, blockBuffer, vectorsInBlock * dim, dim);
        vectorsInBlock++;
        if (vectorsInBlock == blockSize) {
          rawOffsets[blockOrd++] = flushLZ4Block(blockBuffer, vectorsInBlock * dim) - rawDataOffset;
          vectorsInBlock = 0;
        }
      }
    }
    if (vectorsInBlock > 0) {
      rawOffsets[blockOrd++] = flushLZ4Block(blockBuffer, vectorsInBlock * dim) - rawDataOffset;
    }
    
    long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
    long rawDataLength = rawData == null ? 0 : rawData.getFilePointer() - rawDataOffset;
    writeMeta(
        fieldData.fieldInfo, maxDoc, vectorDataOffset, vectorDataLength, rawDataOffset, rawDataLength, fieldData.docsWithField, rawOffsets);
  }

  private long flushLZ4Block(float[] blockBuffer, int floatLength) throws IOException {
    long offset = rawData.getFilePointer();
    byte[] shuffledBytes = new byte[floatLength * 4];
    float[] trimmed = new float[floatLength];
    System.arraycopy(blockBuffer, 0, trimmed, 0, floatLength);
    VectorUtil.byteShuffle(trimmed, shuffledBytes);
    rawData.writeVInt(shuffledBytes.length); // write compressed chunk len metadata? LZ4 doesn't need it but we need decompressed length
    LZ4.compress(shuffledBytes, 0, shuffledBytes.length, rawData, new LZ4.FastCompressionHashTable());
    return offset;
  }

  private void writePackedVector(float[] v, float[] codebook, float[] centroids)
      throws IOException {
    // 1. measure original norm
    float normSq = VectorUtil.dotProduct(v, v);
    float originalNorm = (float) Math.sqrt(normSq);

    // 2. rotate
    float[] rotated = v.clone();
    RotorQuant.rotate(rotated, codebook);

    // 3. quantize to bits indices + get reconstructed norm
    float[] reconNormOut = new float[1];
    byte[] packed = RotorQuant.quantize(rotated, centroids, encoding.bits, reconNormOut);

    // 4. store packed array
    vectorData.writeBytes(packed, packed.length);

    // 5. store scale factor and normSq
    float scale = reconNormOut[0] > 0 ? originalNorm / reconNormOut[0] : 0f;
    vectorData.writeInt(Float.floatToIntBits(scale));
    vectorData.writeInt(Float.floatToIntBits(normSq));

    // 6. store original fp32
    if (compress == IsoCompress.NONE) {
      for (float f : v) {
        vectorData.writeInt(Float.floatToIntBits(f));
      }
    }
  }

  private void writeSortingField(FieldWriter fieldData, int maxDoc, Sorter.DocMap sortMap) throws IOException {
    final int[] ordMap = new int[fieldData.docsWithField.cardinality()];
    DocsWithFieldSet newDocsWithField = new DocsWithFieldSet();
    mapOldOrdToNewOrd(fieldData.docsWithField, sortMap, null, ordMap, newDocsWithField);

    long vectorDataOffset = vectorData.alignFilePointer(64);
    long rawDataOffset = rawData == null ? 0 : rawData.alignFilePointer(64);

    int dim = fieldData.fieldInfo.getVectorDimension();
    float[] codebook = RotorQuant.generateCodebook(seed, dim, variant);
    float[] centroids = RotorQuant.getLloydMaxCentroids(encoding.bits, dim);

    int count = newDocsWithField.cardinality();
    int blockSize = 1 << IsoQuantFlatFormat.LZ4_BLOCK_SHIFT;
    int numBlocks = (count + blockSize - 1) / blockSize;
    long[] rawOffsets = compress != IsoCompress.NONE ? new long[numBlocks] : null;
    int blockOrd = 0;
    
    float[] blockBuffer = compress != IsoCompress.NONE ? new float[blockSize * dim] : null;
    int vectorsInBlock = 0;

    for (int ordinal : ordMap) {
      float[] vector = fieldData.vectors.get(ordinal);
      writePackedVector(vector, codebook, centroids);
      if (compress != IsoCompress.NONE) {
        System.arraycopy(vector, 0, blockBuffer, vectorsInBlock * dim, dim);
        vectorsInBlock++;
        if (vectorsInBlock == blockSize) {
          rawOffsets[blockOrd++] = flushLZ4Block(blockBuffer, vectorsInBlock * dim) - rawDataOffset;
          vectorsInBlock = 0;
        }
      }
    }
    if (vectorsInBlock > 0) {
      rawOffsets[blockOrd++] = flushLZ4Block(blockBuffer, vectorsInBlock * dim) - rawDataOffset;
    }

    long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
    long rawDataLength = rawData == null ? 0 : rawData.getFilePointer() - rawDataOffset;
    writeMeta(fieldData.fieldInfo, maxDoc, vectorDataOffset, vectorDataLength, rawDataOffset, rawDataLength, newDocsWithField, rawOffsets);
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    long vectorDataOffset = vectorData.alignFilePointer(64);
    long rawDataOffset = rawData == null ? 0 : rawData.alignFilePointer(64);

    int dim = fieldInfo.getVectorDimension();
    float[] codebook = RotorQuant.generateCodebook(seed, dim, variant);
    float[] centroids = RotorQuant.getLloydMaxCentroids(encoding.bits, dim);

    DocsWithFieldSet docsWithField = new DocsWithFieldSet();
    int count = 0;
    
    int blockSize = 1 << IsoQuantFlatFormat.LZ4_BLOCK_SHIFT;
    int maxDoc = mergeState.segmentInfo.maxDoc();
    int numBlocks = (maxDoc + blockSize - 1) / blockSize;
    long[] rawOffsets = compress != IsoCompress.NONE ? new long[numBlocks] : null;
    int blockOrd = 0;
    float[] blockBuffer = compress != IsoCompress.NONE ? new float[blockSize * dim] : null;
    int vectorsInBlock = 0;

    List<IsoQuantMergeSub> subs = new ArrayList<>();
    for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
      org.apache.lucene.codecs.KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
      if (reader != null) {
        FloatVectorValues values = reader.getFloatVectorValues(fieldInfo.name);
        if (values != null) {
          boolean fastPath = false;
          if (values instanceof IsoQuantOffHeapVectorValues) {
            IsoQuantOffHeapVectorValues isoValues = (IsoQuantOffHeapVectorValues) values;
            if (isoValues.getIsoEncoding() == encoding && isoValues.getIsoVariant() == variant) {
              fastPath = true;
            }
          }
          subs.add(new IsoQuantMergeSub(mergeState.docMaps[i], values, fastPath));
        }
      }
    }
    org.apache.lucene.index.DocIDMerger<IsoQuantMergeSub> docIdMerger = org.apache.lucene.index.DocIDMerger.of(subs, mergeState.needsIndexSort);

    int bytePackSize = dim * encoding.bits / 8;
    byte[] packedResult = new byte[bytePackSize];
    float[] scaleAndNormResult = new float[2];

    IsoQuantMergeSub current;
    while ((current = docIdMerger.next()) != null) {
      int docV = current.mappedDocID;
      int targetOrd = current.index();

      if (current.fastPath) {
        IsoQuantOffHeapVectorValues isoValues = (IsoQuantOffHeapVectorValues) current.values;
        isoValues.getPackedBytesScaleAndNormSq(targetOrd, packedResult, scaleAndNormResult);
        vectorData.writeBytes(packedResult, packedResult.length);
        vectorData.writeInt(Float.floatToIntBits(scaleAndNormResult[0]));
        vectorData.writeInt(Float.floatToIntBits(scaleAndNormResult[1]));
        if (compress == IsoCompress.NONE) {
          float[] value = current.values.vectorValue(targetOrd);
          for (float v : value) {
            vectorData.writeInt(Float.floatToIntBits(v));
          }
        } else {
          float[] value = current.values.vectorValue(targetOrd);
          System.arraycopy(value, 0, blockBuffer, vectorsInBlock * dim, dim);
          vectorsInBlock++;
          if (vectorsInBlock == blockSize) {
            rawOffsets[blockOrd++] = flushLZ4Block(blockBuffer, vectorsInBlock * dim) - rawDataOffset;
            vectorsInBlock = 0;
          }
        }
      } else {
        float[] value = current.values.vectorValue(targetOrd);
        writePackedVector(value, codebook, centroids);
        if (compress != IsoCompress.NONE) {
          System.arraycopy(value, 0, blockBuffer, vectorsInBlock * dim, dim);
          vectorsInBlock++;
          if (vectorsInBlock == blockSize) {
            rawOffsets[blockOrd++] = flushLZ4Block(blockBuffer, vectorsInBlock * dim) - rawDataOffset;
            vectorsInBlock = 0;
          }
        }
      }
      docsWithField.add(docV);
      count++;
    }
    if (vectorsInBlock > 0) {
      rawOffsets[blockOrd++] = flushLZ4Block(blockBuffer, vectorsInBlock * dim) - rawDataOffset;
    }
    numBlocks = blockOrd; // Update actual used blocks

    if (rawOffsets != null) {
      long[] trimmed = new long[numBlocks];
      System.arraycopy(rawOffsets, 0, trimmed, 0, numBlocks);
      rawOffsets = trimmed;
    }

    long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
    long rawDataLength = rawData == null ? 0 : rawData.getFilePointer() - rawDataOffset;
    writeMeta(
        fieldInfo,
        segmentWriteState.segmentInfo.maxDoc(),
        vectorDataOffset,
        vectorDataLength,
        rawDataOffset,
        rawDataLength,
        docsWithField, rawOffsets);
  }

  @Override
  public CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(
      FieldInfo fieldInfo, MergeState mergeState) throws IOException {

    long vectorDataOffset = vectorData.alignFilePointer(64);

    // First, let's just write everything into a temp file exactly like Lucene99
        org.apache.lucene.store.IndexOutput tempVectorData =
        segmentWriteState.directory.createTempOutput(
            vectorData.getName(), "temp", segmentWriteState.context);

    org.apache.lucene.store.IndexOutput tempRawData = compress != IsoCompress.NONE
        ? segmentWriteState.directory.createTempOutput(rawData == null ? "temp_raw" : rawData.getName(), "temp_raw", segmentWriteState.context)
        : null;

    org.apache.lucene.store.IndexInput vectorDataInput = null;
    org.apache.lucene.store.IndexInput rawDataInput = null;

    try {
      int dim = fieldInfo.getVectorDimension();
      float[] codebook = RotorQuant.generateCodebook(seed, dim, variant);
      float[] centroids = RotorQuant.getLloydMaxCentroids(encoding.bits, dim);

      DocsWithFieldSet docsWithField = new DocsWithFieldSet();

      int maxDoc = mergeState.segmentInfo.maxDoc();
      int blockSize = 1 << IsoQuantFlatFormat.LZ4_BLOCK_SHIFT;
      int numBlocks = (maxDoc + blockSize - 1) / blockSize;
      long[] rawOffsets = compress != IsoCompress.NONE ? new long[numBlocks] : null;
      int blockOrd = 0;
      float[] blockBuffer = compress != IsoCompress.NONE ? new float[blockSize * dim] : null;
      int vectorsInBlock = 0;

      List<IsoQuantMergeSub> subs = new ArrayList<>();
      for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
        org.apache.lucene.codecs.KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
        if (reader != null) {
          FloatVectorValues values = reader.getFloatVectorValues(fieldInfo.name);
          if (values != null) {
            boolean fastPath = false;
            if (values instanceof IsoQuantOffHeapVectorValues) {
              IsoQuantOffHeapVectorValues isoValues = (IsoQuantOffHeapVectorValues) values;
              if (isoValues.getIsoEncoding() == encoding && isoValues.getIsoVariant() == variant) {
                fastPath = true;
              }
            }
            subs.add(new IsoQuantMergeSub(mergeState.docMaps[i], values, fastPath));
          }
        }
      }
      org.apache.lucene.index.DocIDMerger<IsoQuantMergeSub> docIdMerger = org.apache.lucene.index.DocIDMerger.of(subs, mergeState.needsIndexSort);

      int bytePackSize = dim * encoding.bits / 8;
      byte[] packedResult = new byte[bytePackSize];
      float[] scaleAndNormResult = new float[2];

      IsoQuantMergeSub current;
      while ((current = docIdMerger.next()) != null) {
        int docV = current.mappedDocID;
        int targetOrd = current.index();

        if (current.fastPath) {
          IsoQuantOffHeapVectorValues isoValues = (IsoQuantOffHeapVectorValues) current.values;
          isoValues.getPackedBytesScaleAndNormSq(targetOrd, packedResult, scaleAndNormResult);
          tempVectorData.writeBytes(packedResult, packedResult.length);
          tempVectorData.writeInt(Float.floatToIntBits(scaleAndNormResult[0]));
          tempVectorData.writeInt(Float.floatToIntBits(scaleAndNormResult[1]));
          if (compress == IsoCompress.NONE) {
            float[] value = current.values.vectorValue(targetOrd);
            for (float v : value) {
              tempVectorData.writeInt(Float.floatToIntBits(v));
            }
          } else {
            float[] value = current.values.vectorValue(targetOrd);
            System.arraycopy(value, 0, blockBuffer, vectorsInBlock * dim, dim);
            vectorsInBlock++;
            if (vectorsInBlock == blockSize) {
              rawOffsets[blockOrd++] = flushLZ4BlockTemp(tempRawData, blockBuffer, vectorsInBlock * dim);
              vectorsInBlock = 0;
            }
          }
        } else {
          float[] value = current.values.vectorValue(targetOrd);
          float normSq = VectorUtil.dotProduct(value, value);
          float originalNorm = (float) Math.sqrt(normSq);
          float[] rotated = value.clone();
          RotorQuant.rotate(rotated, codebook);
          float[] reconNormOut = new float[1];
          byte[] packed = RotorQuant.quantize(rotated, centroids, encoding.bits, reconNormOut);
          tempVectorData.writeBytes(packed, packed.length);
          float scale = reconNormOut[0] > 0 ? originalNorm / reconNormOut[0] : 0f;
          tempVectorData.writeInt(Float.floatToIntBits(scale));
          tempVectorData.writeInt(Float.floatToIntBits(normSq));
          if (compress == IsoCompress.NONE) {
            for (float v : value) {
              tempVectorData.writeInt(Float.floatToIntBits(v));
            }
          } else {
            System.arraycopy(value, 0, blockBuffer, vectorsInBlock * dim, dim);
            vectorsInBlock++;
            if (vectorsInBlock == blockSize) {
              rawOffsets[blockOrd++] = flushLZ4BlockTemp(tempRawData, blockBuffer, vectorsInBlock * dim);
              vectorsInBlock = 0;
            }
          }
        }

        docsWithField.add(docV);
      }
      if (vectorsInBlock > 0) {
        rawOffsets[blockOrd++] = flushLZ4BlockTemp(tempRawData, blockBuffer, vectorsInBlock * dim);
      }
      numBlocks = blockOrd;
      if (rawOffsets != null) {
        long[] trimmed = new long[numBlocks];
        System.arraycopy(rawOffsets, 0, trimmed, 0, numBlocks);
        rawOffsets = trimmed;
      }

      CodecUtil.writeFooter(tempVectorData);
      IOUtils.close(tempVectorData);
      if (tempRawData != null) {
        CodecUtil.writeFooter(tempRawData);
        IOUtils.close(tempRawData);
      }

      vectorDataInput =
          segmentWriteState.directory.openInput(
              tempVectorData.getName(), org.apache.lucene.store.IOContext.DEFAULT);
      if (tempRawData != null) {
        rawDataInput =
            segmentWriteState.directory.openInput(
                tempRawData.getName(), org.apache.lucene.store.IOContext.DEFAULT);
      }

      // Copy to actual data file
      vectorData.copyBytes(vectorDataInput, vectorDataInput.length() - CodecUtil.footerLength());
      CodecUtil.retrieveChecksum(vectorDataInput);
      long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

      long rawDataOffset = rawData == null ? 0 : rawData.alignFilePointer(64);
      long rawDataLength = 0L;
      if (rawData != null && rawDataInput != null) {
        rawData.copyBytes(rawDataInput, rawDataInput.length() - CodecUtil.footerLength());
        CodecUtil.retrieveChecksum(rawDataInput);
        rawDataLength = rawData.getFilePointer() - rawDataOffset;
      }

      writeMeta(
          fieldInfo,
          segmentWriteState.segmentInfo.maxDoc(),
          vectorDataOffset,
          vectorDataLength,
          rawDataOffset,
          rawDataLength,
          docsWithField, rawOffsets);

      final org.apache.lucene.store.IndexInput finalVectorDataInput = vectorDataInput;
      vectorDataInput = null;

      // Provide random access scorer over the temp input data
      int byteSize = (dim * encoding.bits / 8) + 2 * Float.BYTES + (compress == IsoCompress.NONE ? dim * Float.BYTES : 0);
      IsoQuantOffHeapVectorValues offHeapValues =
          new IsoQuantOffHeapVectorValues.DenseOffHeapVectorValues(
              dim,
              docsWithField.cardinality(),
              finalVectorDataInput,
              byteSize,
              vectorsScorer,
              fieldInfo.getVectorSimilarityFunction(), null, null,
              encoding,
              variant,
              compress,
              seed);

      final RandomVectorScorerSupplier randomVectorScorerSupplier =
          vectorsScorer.getRandomVectorScorerSupplier(
              fieldInfo.getVectorSimilarityFunction(), offHeapValues);

      return new org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier() {
        @Override
        public org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer scorer()
            throws IOException {
          return randomVectorScorerSupplier.scorer();
        }

        @Override
        public org.apache.lucene.util.hnsw.RandomVectorScorerSupplier copy() throws IOException {
          return randomVectorScorerSupplier.copy();
        }

        @Override
        public int totalVectorCount() {
          return docsWithField.cardinality();
        }

        @Override
        public void close() throws IOException {
          IOUtils.close(finalVectorDataInput);
          segmentWriteState.directory.deleteFile(tempVectorData.getName());
          if (compress != IsoCompress.NONE) {
            segmentWriteState.directory.deleteFile(tempRawData.getName());
          }
        }
      };

    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, vectorDataInput, tempVectorData, rawDataInput, tempRawData);
      IOUtils.deleteFilesSuppressingExceptions(
          t, segmentWriteState.directory, tempVectorData.getName());
      if (tempRawData != null) {
        IOUtils.deleteFilesSuppressingExceptions(
            t, segmentWriteState.directory, tempRawData.getName());
      }
      throw t;
    }
  }

  private long flushLZ4BlockTemp(IndexOutput out, float[] blockBuffer, int floatLength) throws IOException {
    long offset = out.getFilePointer();
    byte[] shuffledBytes = new byte[floatLength * 4];
    float[] trimmed = new float[floatLength];
    System.arraycopy(blockBuffer, 0, trimmed, 0, floatLength);
    VectorUtil.byteShuffle(trimmed, shuffledBytes);
    out.writeVInt(shuffledBytes.length);
    LZ4.compress(shuffledBytes, 0, shuffledBytes.length, out, new LZ4.FastCompressionHashTable());
    return offset;
  }

  private void writeMeta(
      FieldInfo field,
      int maxDoc,
      long vectorDataOffset,
      long vectorDataLength,
      long rawDataOffset,
      long rawDataLength,
      DocsWithFieldSet docsWithField,
      long[] rawOffsets)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(field.getVectorSimilarityFunction().ordinal());
    meta.writeVInt(encoding.wireNumber);
    meta.writeVInt(variant.wireNumber);
    meta.writeVInt(compress.wireNumber);
    meta.writeVLong(vectorDataOffset);
    meta.writeVLong(vectorDataLength);
    if (compress != IsoCompress.NONE) {
      meta.writeVLong(rawDataOffset);
      meta.writeVLong(rawDataLength);
    }
    meta.writeVInt(field.getVectorDimension());

    int count = docsWithField.cardinality();
    meta.writeInt(count);
    
    if (rawOffsets != null) {
      int numBlocks = rawOffsets.length;
      meta.writeVInt(numBlocks);
      long startPointer = vectorData.getFilePointer();
      meta.writeLong(startPointer);
      DirectMonotonicWriter offsetsWriter = DirectMonotonicWriter.getInstance(meta, vectorData, numBlocks, DIRECT_MONOTONIC_BLOCK_SHIFT);
      for (long o : rawOffsets) {
        offsetsWriter.add(o);
      }
      offsetsWriter.finish();
      meta.writeLong(vectorData.getFilePointer() - startPointer);
    }
    
    OrdToDocDISIReaderConfiguration.writeStoredMeta(
        DIRECT_MONOTONIC_BLOCK_SHIFT, meta, vectorData, count, maxDoc, docsWithField);
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, vectorData, rawData);
  }

  private static class FieldWriter extends FlatFieldVectorsWriter<float[]> {
    private static final long SHALLOW_RAM_BYTES_USED =
        RamUsageEstimator.shallowSizeOfInstance(FieldWriter.class);
    private final FieldInfo fieldInfo;
    private final int dim;
    private final DocsWithFieldSet docsWithField;
    private final List<float[]> vectors;
    private boolean finished;
    private int lastDocID = -1;

    FieldWriter(FieldInfo fieldInfo) {
      super();
      this.fieldInfo = fieldInfo;
      this.dim = fieldInfo.getVectorDimension();
      this.docsWithField = new DocsWithFieldSet();
      vectors = new ArrayList<>();
    }

    @Override
    public float[] copyValue(float[] value) {
      return ArrayUtil.copyOfSubArray(value, 0, dim);
    }

    @Override
    public void addValue(int docID, float[] vectorValue) throws IOException {
      if (finished) {
        throw new IllegalStateException("already finished, cannot add more values");
      }
      assert docID > lastDocID;
      float[] copy = copyValue(vectorValue);
      docsWithField.add(docID);
      vectors.add(copy);
      lastDocID = docID;
    }

    @Override
    public long ramBytesUsed() {
      long size = SHALLOW_RAM_BYTES_USED;
      if (vectors.size() == 0) return size;
      return size
          + docsWithField.ramBytesUsed()
          + (long) vectors.size()
              * (RamUsageEstimator.NUM_BYTES_OBJECT_REF + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER)
          + (long) vectors.size() * fieldInfo.getVectorDimension() * Float.BYTES;
    }

    @Override
    public List<float[]> getVectors() {
      return vectors;
    }

    @Override
    public DocsWithFieldSet getDocsWithFieldSet() {
      return docsWithField;
    }

    @Override
    public void finish() throws IOException {
      this.finished = true;
    }

    @Override
    public boolean isFinished() {
      return finished;
    }
  }
}
