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
package org.apache.lucene.codecs.lsmvec;

import java.io.IOException;
import java.util.Map;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.OffHeapByteVectorValues;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/**
 * Handles ultra-fast geo-centric topological memory-mapped searching natively over the `.vem` buffer. Relies on the
 * delegate flat reader to physically extract the float vectors.
 */
public final class LsmVecVectorsReader extends KnnVectorsReader {

  public static class FieldEntry {
    public long[] nodeOffsets;
    public long vectorDataOffset;
    public long vectorDataLength;
    public int dimension;
    public int size;
    public OrdToDocDISIReaderConfiguration ordToDoc;
  }

  public FieldEntry getFieldEntry(int fieldNumber) {
    return fields.get(fieldNumber);
  }

  public FieldEntry getFieldEntry(String fieldName) {
    FieldInfo info = segmentReadState.fieldInfos.fieldInfo(fieldName);
    if (info != null) {
      return fields.get(info.number);
    }
    return null;
  }

  public IndexInput getVectorEdgeInput() {
    return vectorEdgeInput.clone();
  }

  private final SegmentReadState segmentReadState;

  private final IndexInput metaInput;
  private final IndexInput vectorEdgeInput;
  private final IndexInput vectorDataInput;
  private final Map<Integer, FieldEntry> fields = new java.util.HashMap<>();
  private final FlatVectorsScorer flatVectorsScorer = new DefaultFlatVectorScorer();

  public LsmVecVectorsReader(SegmentReadState state) throws IOException {
    this.segmentReadState = state;

    metaInput =
        state.directory.openInput(
            state.segmentInfo.name + "." + LsmVecVectorsFormat.VEM_META_EXTENSION, state.context);

    boolean success = false;
    try {
      CodecUtil.checkIndexHeader(
          metaInput, "LsmVecMeta", 1, 1, state.segmentInfo.getId(), state.segmentSuffix);

      vectorEdgeInput =
          state.directory.openInput(
              state.segmentInfo.name + "." + LsmVecVectorsFormat.VEM_EXTENSION, state.context);

      CodecUtil.checkIndexHeader(
          vectorEdgeInput, "LsmVecEdge", 1, 1, state.segmentInfo.getId(), state.segmentSuffix);

      vectorDataInput =
          state.directory.openInput(
              state.segmentInfo.name + "." + LsmVecVectorsFormat.VECD_EXTENSION, state.context);

      CodecUtil.checkIndexHeader(
          vectorDataInput, "LsmVecData", 1, 1, state.segmentInfo.getId(), state.segmentSuffix);

      // Read structural metadata pointers per field mapping here
      long metaLength = metaInput.length();
      long metaFooterLength = CodecUtil.footerLength();
      while (metaInput.getFilePointer() < metaLength - metaFooterLength) {
        int fieldNumber = metaInput.readInt();
        FieldEntry entry = new FieldEntry();
        metaInput.readLong(); // Read and ignore edgePtr
        entry.vectorDataOffset = metaInput.readVLong();
        entry.vectorDataLength = metaInput.readVLong();

        int totalOrds = metaInput.readInt();
        entry.nodeOffsets = new long[totalOrds];
        for (int i = 0; i < totalOrds; i++) {
          entry.nodeOffsets[i] = metaInput.readLong();
        }

        entry.dimension = metaInput.readVInt();
        entry.size = metaInput.readInt();
        entry.ordToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(metaInput, entry.size);
        fields.put(fieldNumber, entry);
      }

      success = true;
    } finally {
      if (!success) {
        if (metaInput != null) {
          metaInput.close();
        }
      }
    }
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo(field);
    if (fieldInfo == null || !fields.containsKey(fieldInfo.number)) return null;
    FieldEntry entry = fields.get(fieldInfo.number);
    return OffHeapFloatVectorValues.load(
        fieldInfo.getVectorSimilarityFunction(),
        flatVectorsScorer,
        entry.ordToDoc,
        fieldInfo.getVectorEncoding(),
        entry.dimension,
        entry.vectorDataOffset,
        entry.vectorDataLength,
        vectorDataInput.clone());
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo(field);
    if (fieldInfo == null || !fields.containsKey(fieldInfo.number)) return null;
    FieldEntry entry = fields.get(fieldInfo.number);
    return OffHeapByteVectorValues.load(
        fieldInfo.getVectorSimilarityFunction(),
        flatVectorsScorer,
        entry.ordToDoc,
        fieldInfo.getVectorEncoding(),
        entry.dimension,
        entry.vectorDataOffset,
        entry.vectorDataLength,
        vectorDataInput.clone());
  }

  public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
    FloatVectorValues values = getFloatVectorValues(field);
    if (values == null) return null;
    return flatVectorsScorer.getRandomVectorScorer(
        segmentReadState.fieldInfos.fieldInfo(field).getVectorSimilarityFunction(),
        values,
        target);
  }

  public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
    ByteVectorValues values = getByteVectorValues(field);
    if (values == null) return null;
    return flatVectorsScorer.getRandomVectorScorer(
        segmentReadState.fieldInfos.fieldInfo(field).getVectorSimilarityFunction(),
        values,
        target);
  }

  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    RandomVectorScorer scorer = getRandomVectorScorer(field, target);
    if (scorer == null) return;
    FloatVectorValues values = getFloatVectorValues(field);
    if (values == null) return;
    doSearch(field, scorer, knnCollector, acceptDocs, values::ordToDoc);
  }

  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    RandomVectorScorer scorer = getRandomVectorScorer(field, target);
    if (scorer == null) return;
    ByteVectorValues values = getByteVectorValues(field);
    if (values == null) return;
    doSearch(field, scorer, knnCollector, acceptDocs, values::ordToDoc);
  }

  private void doSearch(
      String field,
      RandomVectorScorer scorer,
      KnnCollector knnCollector,
      AcceptDocs acceptDocs,
      java.util.function.IntUnaryOperator ordToDoc)
      throws IOException {
    FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo(field);
    long[] nodeOffsets = fields.get(fieldInfo.number).nodeOffsets;
    if (nodeOffsets == null) return;

    int N = segmentReadState.segmentInfo.maxDoc();
    int efSearch = (int) (knnCollector.k() * 1.5);

    org.apache.lucene.util.SparseFixedBitSet visited =
        new org.apache.lucene.util.SparseFixedBitSet(N);
    org.apache.lucene.util.hnsw.NeighborQueue candidates =
        new org.apache.lucene.util.hnsw.NeighborQueue(efSearch, true);
    IndexInput edgeInput = vectorEdgeInput.clone();
    Bits acceptBits = acceptDocs == null ? null : acceptDocs.bits();

    int entryOrd = 0;
    int entryDoc = ordToDoc.applyAsInt(entryOrd);
    if (entryDoc == -1) return;

    float entryScore = scorer.score(entryOrd);
    visited.set(entryDoc);

    float minAcceptedSimilarity = Float.NEGATIVE_INFINITY;

    candidates.add(entryOrd, entryScore);
    if (acceptBits == null || acceptBits.get(entryDoc)) {
      if (knnCollector.collect(entryDoc, entryScore)) {
        minAcceptedSimilarity = knnCollector.minCompetitiveSimilarity();
      }
    }

    // Graph Navigation Layer
    while (candidates.size() > 0 && !knnCollector.earlyTerminated()) {
      float currentScore = candidates.topScore();
      int topNodeOrd = candidates.pop();

      if (currentScore < minAcceptedSimilarity) {
        break;
      }

      edgeInput.seek(nodeOffsets[topNodeOrd]);
      int numNeighbors = edgeInput.readVInt();
      for (int i = 0; i < numNeighbors; i++) {
        int neighborOrd = edgeInput.readVInt();
        int nDoc = ordToDoc.applyAsInt(neighborOrd);
        if (!visited.getAndSet(nDoc)) {
          knnCollector.incVisitedCount(1);
          float s = scorer.score(neighborOrd);

          if (s >= minAcceptedSimilarity) {
            candidates.add(neighborOrd, s);
            if (acceptBits == null || acceptBits.get(nDoc)) {
              if (knnCollector.collect(nDoc, s)) {
                minAcceptedSimilarity = knnCollector.minCompetitiveSimilarity();
              }
            }
          }
        }
      }
    }
  }

  @Override
  public void checkIntegrity() throws IOException {
    CodecUtil.checksumEntireFile(vectorDataInput);
    CodecUtil.checksumEntireFile(vectorEdgeInput);
    CodecUtil.checksumEntireFile(metaInput);
  }

  @Override
  public void close() throws IOException {
    org.apache.lucene.util.IOUtils.close(metaInput, vectorEdgeInput, vectorDataInput);
  }
}
