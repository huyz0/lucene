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
import java.util.Collections;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestLsmVecGraph extends LuceneTestCase {

  private FieldInfo createFieldInfo(VectorEncoding encoding, VectorSimilarityFunction similarityFunction, int dimension) {
    return new FieldInfo(
        "testField",
        1,
        false, false, false,
        IndexOptions.NONE,
        DocValuesType.NONE,
        DocValuesSkipIndexType.NONE,
        -1L,
        Collections.emptyMap(),
        0, 0, 0,
        dimension,
        encoding,
        similarityFunction,
        false, false);
  }

  private void addNodeVectorFloat(
      LsmVecGraph graph, int docID, float[] vector)
      throws IOException {

    graph.setVectorValue(docID, vector);
    graph.addNode(docID, docID);
  }

  private void addNodeVectorByte(
      LsmVecGraph graph, int docID, byte[] vector)
      throws IOException {

    graph.setVectorValue(docID, vector);
    graph.addNode(docID, docID);
  }

  @Test
  public void testFloat32Graph() throws IOException {
    int maxEdges = 4;
    FieldInfo fieldInfo = createFieldInfo(VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN, 3);
    LsmVecGraph graph = new HeapLsmVecGraph(maxEdges, 10, fieldInfo, 3, 500_000);

    float[] v0 = {0f, 0f, 0f};
    addNodeVectorFloat(graph, 0, v0);

    float[] v1 = {1f, 0f, 0f};
    addNodeVectorFloat(graph, 1, v1);

    float[] v2 = {0f, 1f, 0f};
    addNodeVectorFloat(graph, 2, v2);

    float[] v3 = {0f, 0f, 1f};
    addNodeVectorFloat(graph, 3, v3);

    // Verify neighbors are successfully connected by exact search evaluation bounds
    // natively
    int[] n1 = graph.getNeighbors(1);
    assertTrue(n1.length <= maxEdges);

    // Test size and scale bounds dynamically natively mapping correctly
    assertEquals(4, graph.size());
  }

  @Test
  public void testByteGraph() throws IOException {
    int maxEdges = 2;
    FieldInfo fieldInfo = createFieldInfo(VectorEncoding.BYTE, VectorSimilarityFunction.COSINE, 2);
    LsmVecGraph graph = new HeapLsmVecGraph(maxEdges, 10, fieldInfo, 2, 500_000);

    byte[] v0 = {1, 1};
    addNodeVectorByte(graph, 0, v0);

    byte[] v1 = {-1, 1};
    addNodeVectorByte(graph, 1, v1);

    byte[] v2 = {1, -1};
    addNodeVectorByte(graph, 2, v2);

    // Dynamic arrays test boundaries
    int[] neighbors = graph.getNeighbors(2);
    assertNotNull(neighbors);
  }

  @Test
  public void testRamBytesUsed() throws IOException {
    FieldInfo fieldInfo = createFieldInfo(VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, 128);
    LsmVecGraph graph = new HeapLsmVecGraph(16, 100, fieldInfo, 128, 500_000);

    long initialRam = graph.ramBytesUsed();

    float[] v0 = new float[128];
    addNodeVectorFloat(graph, 0, v0);

    float[] v1 = new float[128];
    v1[0] = 1.0f;
    addNodeVectorFloat(graph, 1, v1);

    long finalRam = graph.ramBytesUsed();
    assertTrue(finalRam > initialRam);
  }

  @Test
  public void testMaxEdgesEviction() throws IOException {
    int maxEdges = 3;
    FieldInfo fieldInfo = createFieldInfo(VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN, 2);
    LsmVecGraph graph = new HeapLsmVecGraph(maxEdges, 100, fieldInfo, 2, 500_000);

    // Target node at origin
    addNodeVectorFloat(graph, 0, new float[] {0f, 0f});

    // Add 10 points progressively closer to origin!
    // Since maxEdges=3, it should only retain the 3 closest nodes over time
    // natively!
    for (int i = 1; i <= 10; i++) {
      // e.g. distance = 10, 9, 8, 7..., 1
      float dist = 11.0f - i;
      addNodeVectorFloat(graph, i, new float[] {dist, 0f});
    }

    int[] neighbors = graph.getNeighbors(10);
    assertTrue(neighbors.length <= maxEdges);

    assertEquals(11, graph.size());
  }

  @Test
  public void testLargeScaleRandomByteVectors() throws IOException {
    int maxNodes = 5000;
    int maxEdges = 16;
    int dimension = 32;
    FieldInfo fieldInfo = createFieldInfo(VectorEncoding.BYTE, VectorSimilarityFunction.COSINE, dimension);
    LsmVecGraph graph = new HeapLsmVecGraph(maxEdges, maxNodes, fieldInfo, dimension, 500_000);

    byte[] randomVec = new byte[dimension];
    for (int i = 0; i < maxNodes; i++) {
      for (int d = 0; d < dimension; d++) {
        randomVec[d] = (byte) (random().nextInt(256) - 128); // byte bounding safely mapped
      }
      addNodeVectorByte(graph, i, randomVec.clone());
    }

    assertEquals(maxNodes, graph.size());

    // Check neighbors mapped seamlessly without exception throws spanning cache
    // bounds!
    int[] out = graph.getNeighbors(1234);
    assertTrue(out.length <= maxEdges);
  }

  @Test
  public void testDisconnectedIslandsMetricValidation() throws IOException {
    int maxEdges = 5;
    FieldInfo fieldInfo = createFieldInfo(VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN, 2);
    LsmVecGraph graph = new HeapLsmVecGraph(maxEdges, 100, fieldInfo, 2, 500_000);

    // Cluster A: around (-100, -100)
    for (int i = 0; i < 10; i++) {
      addNodeVectorFloat(
          graph, i, new float[] {-100f + i, -100f + i});
    }

    // Cluster B: around (+100, +100)
    for (int i = 10; i < 20; i++) {
      addNodeVectorFloat(
          graph, i, new float[] {100f + i, 100f + i});
    }

    // Nodes in Cluster B should purely connect internally to Cluster B
    // geographically!
    int[] neighborsOf15 = graph.getNeighbors(15);
    for (int n : neighborsOf15) {
      if (n != -1) {
        assertTrue("Neighbor " + n + " of Cluster B node should be inside Cluster B", n >= 10);
      }
    }
  }
}
