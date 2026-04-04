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

import java.io.IOException;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.util.compress.LZ4;
import org.apache.lucene.util.VectorUtil;

/** Read the vector values from the index input. This supports both iterated and random access. */
public abstract class IsoQuantOffHeapVectorValues extends FloatVectorValues {

  protected final int dimension;
  protected final int size;
  protected final IndexInput slice;
  protected final int byteSize;
  protected int lastOrd = -1;
  protected final float[] value;
  protected final IndexInput rawData;
  protected final DirectMonotonicReader rawOffsetsReader;
  protected final IsoCompress compress;
  protected final byte[] decompressedBytes;
  protected int lastBlockOrd = -1;
  protected final float[] blockBufferCache;
  protected final VectorSimilarityFunction similarityFunction;
  protected final FlatVectorsScorer flatVectorsScorer;

  protected final long seed;
  protected final IsoEncoding encoding;
  protected final IsoVariant variant;
  protected final float[] codebook;
  protected final float[] centroids;
  protected final byte[] packedBuffer;

  IsoQuantOffHeapVectorValues(
      int dimension,
      int size,
      IndexInput slice,
      int byteSize,
      FlatVectorsScorer flatVectorsScorer,
      VectorSimilarityFunction similarityFunction,
      IndexInput rawData,
      DirectMonotonicReader rawOffsetsReader,
      IsoEncoding encoding,
      IsoVariant variant,
      IsoCompress compress,
      long seed) {
    this.dimension = dimension;
    this.size = size;
    this.slice = slice;
    this.byteSize = byteSize;
    this.similarityFunction = similarityFunction;
    this.flatVectorsScorer = flatVectorsScorer;
    this.rawData = rawData;
    this.rawOffsetsReader = rawOffsetsReader;
    this.encoding = encoding;
    this.variant = variant;
    this.compress = compress;
    this.seed = seed;
    this.value = new float[dimension];
    int blockSize = 1 << IsoQuantFlatFormat.LZ4_BLOCK_SHIFT;
    this.decompressedBytes = compress == IsoCompress.LZ4 ? new byte[blockSize * dimension * 4] : null;
    this.blockBufferCache = compress == IsoCompress.LZ4 ? new float[blockSize * dimension] : null;
    this.packedBuffer = new byte[dimension * encoding.bits / 8];
    this.codebook =
        org.apache.lucene.util.RotorQuant.generateInverseCodebook(
            org.apache.lucene.util.RotorQuant.generateCodebook(seed, dimension, variant));
    this.centroids = org.apache.lucene.util.RotorQuant.getLloydMaxCentroids(encoding.bits, dimension);
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public int size() {
    return size;
  }

  public IsoEncoding getIsoEncoding() {
    return encoding;
  }

  public IsoVariant getIsoVariant() {
    return variant;
  }

  public IndexInput getSlice() {
    return slice;
  }

  @Override
  public float[] vectorValue(int targetOrd) throws IOException {
    if (lastOrd == targetOrd) {
      return value;
    }

    if (compress == IsoCompress.NONE) {
      long offset = (long) targetOrd * byteSize;
      // Skip packed bytes, scale, and normSq
      slice.seek(offset + (dimension * encoding.bits / 8) + 8);
      for (int i = 0; i < dimension; i++) {
        value[i] = Float.intBitsToFloat(slice.readInt());
      }
    } else {
      int blockOrd = targetOrd >> IsoQuantFlatFormat.LZ4_BLOCK_SHIFT;
      if (blockOrd != lastBlockOrd) {
        long offset = rawOffsetsReader.get(blockOrd);
        rawData.seek(offset);
        int decompressedLen = rawData.readVInt();
        LZ4.decompress(rawData, decompressedLen, decompressedBytes, 0);
        int floatsLen = decompressedLen / 4;
        if (floatsLen == blockBufferCache.length) {
            VectorUtil.byteUnshuffle(decompressedBytes, blockBufferCache);
        } else {
            byte[] exactSource = new byte[decompressedLen];
            System.arraycopy(decompressedBytes, 0, exactSource, 0, decompressedLen);
            float[] exactDest = new float[floatsLen];
            VectorUtil.byteUnshuffle(exactSource, exactDest);
            System.arraycopy(exactDest, 0, blockBufferCache, 0, floatsLen);
        }
        lastBlockOrd = blockOrd;
      }
      int inBlockOrd = targetOrd & ((1 << IsoQuantFlatFormat.LZ4_BLOCK_SHIFT) - 1);
      System.arraycopy(blockBufferCache, inBlockOrd * dimension, value, 0, dimension);
    }

    lastOrd = targetOrd;
    return value;
  }

  public void getPackedBytesScaleAndNormSq(int targetOrd, byte[] packedResult, float[] scaleAndNormResult)
      throws IOException {
    slice.seek((long) targetOrd * byteSize);
    slice.readBytes(packedResult, 0, packedResult.length);
    scaleAndNormResult[0] = Float.intBitsToFloat(slice.readInt()); // scale
    scaleAndNormResult[1] = Float.intBitsToFloat(slice.readInt()); // normSq
  }

  public void prefetch(final int[] ordsToPrefetch, int numOrds) throws IOException {
    if (ordsToPrefetch == null) {
      return;
    }

    int finalNumOrds = Math.min(numOrds, ordsToPrefetch.length);
    if (finalNumOrds <= 1) {
      return;
    }

    // 1. calculate offset and prefetch immediately
    for (int i = 0; i < numOrds; i++) {
      long offset = (long) ordsToPrefetch[i] * byteSize;
      slice.prefetch(offset, byteSize);
    }
  }

  public static IsoQuantOffHeapVectorValues load(
      VectorSimilarityFunction vectorSimilarityFunction,
      FlatVectorsScorer flatVectorsScorer,
      OrdToDocDISIReaderConfiguration configuration,
      VectorEncoding vectorEncoding,
      int dimension,
      long vectorDataOffset,
      long vectorDataLength,
      IndexInput vectorData,
      IndexInput rawData,
      DirectMonotonicReader rawOffsetsReader,
      IsoEncoding encoding,
      IsoVariant variant,
      IsoCompress compress,
      long seed,
      int size)
      throws IOException {
    if (compress == IsoCompress.ZSTD || compress == IsoCompress.LZAV) {
      throw new UnsupportedOperationException(compress + " requires external SPI implementation and is not natively supported in lucene/core");
    }
    if (configuration.isEmpty() || vectorEncoding != VectorEncoding.FLOAT32) {
      return new EmptyOffHeapVectorValues(
          dimension, flatVectorsScorer, vectorSimilarityFunction, encoding, variant, compress, seed);
    }
    IndexInput bytesSlice = vectorData.slice("vector-data", vectorDataOffset, vectorDataLength);
    // byteSize = packed indices + scale + normSq + optionally original fp32
    int byteSize = (dimension * encoding.bits / 8) + 8 + (compress == IsoCompress.NONE ? (dimension * Float.BYTES) : 0);
    if (configuration.isDense()) {
      return new DenseOffHeapVectorValues(
          dimension,
          size,
          bytesSlice,
          byteSize,
          flatVectorsScorer,
          vectorSimilarityFunction,
          rawData,
          rawOffsetsReader,
          encoding,
          variant,
          compress,
          seed);
    } else {
      return new SparseOffHeapVectorValues(
          configuration,
          vectorData,
          bytesSlice,
          dimension,
          byteSize,
          flatVectorsScorer,
          vectorSimilarityFunction,
          rawData,
          rawOffsetsReader,
          encoding,
          variant,
          compress,
          seed,
          size);
    }
  }

  /**
   * Dense vector values that are stored off-heap. This is the most common case when every doc has a
   * vector.
   */
  public static class DenseOffHeapVectorValues extends IsoQuantOffHeapVectorValues {

    public DenseOffHeapVectorValues(
        int dimension,
        int size,
        IndexInput slice,
        int byteSize,
        FlatVectorsScorer flatVectorsScorer,
        VectorSimilarityFunction similarityFunction,
        IndexInput rawData,
        DirectMonotonicReader rawOffsetsReader,
        IsoEncoding encoding,
        IsoVariant variant,
        IsoCompress compress,
        long seed) {
      super(dimension, size, slice, byteSize, flatVectorsScorer, similarityFunction, rawData, rawOffsetsReader, encoding, variant, compress, seed);
    }

    @Override
    public DenseOffHeapVectorValues copy() throws IOException {
      return new DenseOffHeapVectorValues(
          dimension,
          size,
          slice.clone(),
          byteSize,
          flatVectorsScorer,
          similarityFunction,
          rawData != null ? rawData.clone() : null,
          rawOffsetsReader,
          encoding,
          variant,
          compress,
          seed);
    }

    @Override
    public int ordToDoc(int ord) {
      return ord;
    }

    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
      return acceptDocs;
    }

    @Override
    public DocIndexIterator iterator() {
      return createDenseIterator();
    }

    @Override
    public VectorScorer scorer(float[] query) throws IOException {
      DenseOffHeapVectorValues copy = copy();
      DocIndexIterator iterator = copy.iterator();
      RandomVectorScorer randomVectorScorer =
          flatVectorsScorer.getRandomVectorScorer(similarityFunction, copy, query);
      return new VectorScorer() {
        @Override
        public float score() throws IOException {
          return randomVectorScorer.score(iterator.docID());
        }

        @Override
        public DocIdSetIterator iterator() {
          return iterator;
        }

        @Override
        public VectorScorer.Bulk bulk(DocIdSetIterator matchingDocs) {
          return Bulk.fromRandomScorerDense(randomVectorScorer, iterator, matchingDocs);
        }
      };
    }
  }

  private static class SparseOffHeapVectorValues extends IsoQuantOffHeapVectorValues {
    private final DirectMonotonicReader ordToDoc;
    private final IndexedDISI disi;
    // dataIn was used to init a new IndexedDIS for #randomAccess()
    private final IndexInput dataIn;
    private final OrdToDocDISIReaderConfiguration configuration;

    public SparseOffHeapVectorValues(
        OrdToDocDISIReaderConfiguration configuration,
        IndexInput dataIn,
        IndexInput slice,
        int dimension,
        int byteSize,
        FlatVectorsScorer flatVectorsScorer,
        VectorSimilarityFunction similarityFunction,
        IndexInput rawData,
        DirectMonotonicReader rawOffsetsReader,
        IsoEncoding encoding,
        IsoVariant variant,
        IsoCompress compress,
        long seed,
        int size)
        throws IOException {

      super(dimension, size, slice, byteSize, flatVectorsScorer, similarityFunction, rawData, rawOffsetsReader, encoding, variant, compress, seed);
      this.configuration = configuration;
      this.dataIn = dataIn;
      this.ordToDoc = configuration.getDirectMonotonicReader(dataIn);
      this.disi = configuration.getIndexedDISI(dataIn);
    }

    @Override
    public SparseOffHeapVectorValues copy() throws IOException {
      return new SparseOffHeapVectorValues(
          configuration,
          dataIn.clone(), // needed to fix clone missing
          slice.clone(),
          dimension,
          byteSize,
          flatVectorsScorer,
          similarityFunction,
          rawData != null ? rawData.clone() : null,
          rawOffsetsReader,
          encoding,
          variant,
          compress,
          seed,
          size);
    }

    @Override
    public int ordToDoc(int ord) {
      return (int) ordToDoc.get(ord);
    }

    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
      if (acceptDocs == null) {
        return null;
      }
      return new Bits() {
        @Override
        public boolean get(int index) {
          return acceptDocs.get(ordToDoc(index));
        }

        @Override
        public int length() {
          return size;
        }
      };
    }

    @Override
    public DocIndexIterator iterator() {
      return IndexedDISI.asDocIndexIterator(disi);
    }

    @Override
    public VectorScorer scorer(float[] query) throws IOException {
      SparseOffHeapVectorValues copy = copy();
      DocIndexIterator iterator = copy.iterator();
      RandomVectorScorer randomVectorScorer =
          flatVectorsScorer.getRandomVectorScorer(similarityFunction, copy, query);
      return new VectorScorer() {
        @Override
        public float score() throws IOException {
          return randomVectorScorer.score(iterator.index());
        }

        @Override
        public DocIdSetIterator iterator() {
          return iterator;
        }

        @Override
        public VectorScorer.Bulk bulk(DocIdSetIterator matchingDocs) {
          return Bulk.fromRandomScorerSparse(randomVectorScorer, iterator, matchingDocs);
        }
      };
    }
  }

  private static class EmptyOffHeapVectorValues extends IsoQuantOffHeapVectorValues {

    public EmptyOffHeapVectorValues(
        int dimension,
        FlatVectorsScorer flatVectorsScorer,
        VectorSimilarityFunction similarityFunction,
        IsoEncoding encoding,
        IsoVariant variant,
        IsoCompress compress,
        long seed) {
      super(dimension, 0, null, 0, flatVectorsScorer, similarityFunction, null, null, encoding, variant, compress, seed);
    }

    @Override
    public int dimension() {
      return super.dimension();
    }

    @Override
    public int size() {
      return 0;
    }

    @Override
    public EmptyOffHeapVectorValues copy() {
      throw new UnsupportedOperationException();
    }

    @Override
    public float[] vectorValue(int targetOrd) {
      throw new UnsupportedOperationException();
    }

    @Override
    public DocIndexIterator iterator() {
      return createDenseIterator();
    }

    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
      return null;
    }

    @Override
    public VectorScorer scorer(float[] query) {
      return null;
    }
  }
}
