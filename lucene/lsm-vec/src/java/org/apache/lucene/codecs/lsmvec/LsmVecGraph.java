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
import java.util.Arrays;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;

/**
 * An ephemeral, in-memory representation of the LSM-VEC graph structure.
 *
 * <p>Stores the graph adjacency list entirely using primal `int[][]` matrices avoiding any Object
 * bloat. Maps Document IDs strictly to an array of connected neighbor Document IDs.
 */
public abstract class LsmVecGraph {
  protected final FieldInfo fieldInfo;
  protected final VectorSimilarityFunction similarityFunction;
  protected final VectorEncoding encoding;
  protected final int vectorDimension;
  protected final int vectorsPerBlock;
  private final int efConstruction;
  private final NeighborQueue candidates;
  private final NeighborQueue results;
  private final int[] sortedNodesBuffer;
  private final float[] sortedScoresBuffer;
  private final int[] selectedBuffer;
  private final float[] selectedScoresBuffer;
  protected final int[] scratchBulkNodes;
  protected final float[] scratchBulkScores;

  // A 1D flattened array spanning maxDocs * maxEdges exclusively without headers!
  private int[] nodes;
  private float[] edgeScores;
  private int size;
  private final int maxEdges;
  private int maxDoc;
  private int entryPoint = -1;
  private int[] visitedNodes;
  private int visitVersion = 0;

  public LsmVecGraph(int maxEdges, int efConstruction, FieldInfo fieldInfo, int vectorDimension, int vectorsPerBlock) {
    this.fieldInfo = fieldInfo;
    this.similarityFunction = fieldInfo.getVectorSimilarityFunction();
    this.encoding = fieldInfo.getVectorEncoding();
    this.vectorsPerBlock = vectorsPerBlock;
    this.vectorDimension = vectorDimension;
    this.nodes = new int[0];
    this.edgeScores = new float[0];
    this.maxEdges = maxEdges;
    this.efConstruction = efConstruction;
    this.size = 0;

    this.candidates = new NeighborQueue(efConstruction, true);
    this.results = new NeighborQueue(efConstruction, false);
    // +1 needed since results.insertWithOverflow might exceed efConstruction slightly before trim
    int queueCap = efConstruction + maxEdges + 1;
    this.sortedNodesBuffer = new int[queueCap];
    this.sortedScoresBuffer = new float[queueCap];
    this.selectedBuffer = new int[maxEdges];
    this.selectedScoresBuffer = new float[maxEdges];
    this.visitedNodes = new int[0];
    this.scratchBulkNodes = new int[maxEdges];
    this.scratchBulkScores = new float[maxEdges];
  }

  public abstract float computeDistance(int aOrd, int bOrd);

  public abstract void bulkComputeDistance(int targetOrd, int[] neighborOrds, float[] outScores, int count);

  public abstract void setVectorValue(int targetOrd, Object vectorValue);

  public abstract void writeVectorData(
      int ord,
      org.apache.lucene.store.IndexOutput vectorDataOutput,
      java.nio.FloatBuffer floatBuffer,
      byte[] floatBufferBytes)
      throws java.io.IOException;

  /**
   * Evaluates scores via an explicit RandomVectorScorer dynamically yielding distances globally natively.
   */
  public void addNode(int docID, int targetOrd) throws IOException {
    addNodeWithoutSearch(docID);
    buildNeighbors(docID, targetOrd, null);
  }

  public void addNodeWithHints(int docID, int targetOrd, int[] hintOrds) throws IOException {
    addNodeWithoutSearch(docID);
    buildNeighbors(docID, targetOrd, hintOrds);
  }

  public void addNodeWithoutSearch(int docID) {
    addNode(docID);

    if (docID >= maxDoc) {
      maxDoc = docID + 1;
    }

    if (entryPoint == -1) {
      entryPoint = docID;
    }
  }

  private void buildNeighbors(int docID, int targetOrd, int[] hintOrds) throws IOException {
    if (visitVersion == Integer.MAX_VALUE) {
      Arrays.fill(visitedNodes, 0);
      visitVersion = 0;
    }
    visitVersion++;
    candidates.clear();
    results.clear();

    float minAcceptedSimilarity = Float.NEGATIVE_INFINITY;

    if (hintOrds != null) {
      for (int hintOrd : hintOrds) {
        if (hintOrd == -1 || hintOrd == targetOrd) continue;
        float score = computeDistance(targetOrd, hintOrd);
        if (hintOrd < visitedNodes.length) {
          visitedNodes[hintOrd] = visitVersion;
        }
        candidates.add(hintOrd, score);
        results.insertWithOverflow(hintOrd, score);
      }
    }

    if (entryPoint != -1 && entryPoint != targetOrd && (entryPoint >= visitedNodes.length
        || visitedNodes[entryPoint] != visitVersion)) {
      float entryScore = computeDistance(targetOrd, entryPoint);
      if (entryPoint < visitedNodes.length) {
        visitedNodes[entryPoint] = visitVersion;
      }
      candidates.add(entryPoint, entryScore);
      results.insertWithOverflow(entryPoint, entryScore);
    }

    if (results.size() >= efConstruction) {
      minAcceptedSimilarity = results.topScore();
    }

    while (candidates.size() > 0) {
      float currentScore = candidates.topScore();
      int topNodeOrd = candidates.pop();

      if (currentScore < minAcceptedSimilarity) {
        break;
      }

      int offset = topNodeOrd * maxEdges;
      if (offset >= nodes.length) continue;

      int bulkCount = 0;
      for (int i = 0; i < maxEdges; i++) {
        int neighborDoc = nodes[offset + i];
        if (neighborDoc == -1) continue;

        if (neighborDoc < visitedNodes.length && visitedNodes[neighborDoc] != visitVersion) {
          visitedNodes[neighborDoc] = visitVersion;
          scratchBulkNodes[bulkCount++] = neighborDoc;
        }
      }

      if (bulkCount > 0) {
        bulkComputeDistance(targetOrd, scratchBulkNodes, scratchBulkScores, bulkCount);
        for (int i = 0; i < bulkCount; i++) {
          int neighborDoc = scratchBulkNodes[i];
          float s = scratchBulkScores[i];
          if (results.size() < efConstruction || s > results.topScore()) {
            candidates.add(neighborDoc, s);
            results.insertWithOverflow(neighborDoc, s);
            if (results.size() >= efConstruction) {
              minAcceptedSimilarity = results.topScore();
            }
          }
        }
      }
    }

    // Vamana Alpha Pruning (Diverse Neighbor Selection)
    int resultSize = results.size();
    // MinHeap pops worst nodes first, so write them backward to get Best -> Worst
    for (int i = resultSize - 1; i >= 0; i--) {
      sortedScoresBuffer[i] = results.topScore();
      sortedNodesBuffer[i] = results.pop();
    }

    int selectedCount = 0;

    for (int i = 0; i < resultSize; i++) {
      int cand = sortedNodesBuffer[i];
      float candScore = sortedScoresBuffer[i];

      boolean isDiverse = true;
      if (selectedCount > 0) {
        System.arraycopy(selectedBuffer, 0, scratchBulkNodes, 0, selectedCount);
        bulkComputeDistance(cand, scratchBulkNodes, scratchBulkScores, selectedCount);
        float candToDoc = candScore;
        for (int j = 0; j < selectedCount; j++) {
          float candToSel = scratchBulkScores[j];
          if (candToSel >= candToDoc) {
            isDiverse = false;
            break;
          }
        }
      }

      if (isDiverse) {
        selectedBuffer[selectedCount] = cand;
        selectedScoresBuffer[selectedCount] = candScore;
        selectedCount++;
        if (selectedCount == maxEdges) break;
      }
    }

    // Add diversified navigated edges!
    for (int i = 0; i < selectedCount; i++) {
      addEdge(docID, selectedBuffer[i], selectedScoresBuffer[i]);
    }
  }

  /** Initializes or fetches a node array for the specified document up to maxEdges capacity. */
  public void addNode(int docID) {
    if (docID >= visitedNodes.length) {
      visitedNodes = ArrayUtil.grow(visitedNodes, docID + 1);
    }
    if ((docID + 1) * maxEdges > nodes.length) {
      int prevLen = nodes.length;
      nodes = ArrayUtil.grow(nodes, (docID + 1) * maxEdges);
      Arrays.fill(nodes, prevLen, nodes.length, -1);
      // grow exact so dimensions align across the 1D space
      float[] newEdgeScores = new float[nodes.length];
      System.arraycopy(edgeScores, 0, newEdgeScores, 0, edgeScores.length);
      edgeScores = newEdgeScores;
    }
    if (docID >= size) {
      size = docID + 1;
    }
  }

  /**
   * Links a bidirectional edge between a target node and a new neighbor. Evicts geographically furthest linked edges if
   * bounds are hit structurally.
   */
  public void addEdge(int docID, int neighborDocID, float score) {
    addNodeWithoutSearch(docID);
    addNodeWithoutSearch(neighborDocID);

    insertEdge(docID, neighborDocID, score);
    insertEdge(neighborDocID, docID, score);
  }

  public void injectEdge(int docID, int neighborDocID, float score) {
    addEdge(docID, neighborDocID, score);
  }

  private void insertEdge(int srcId, int dstId, float score) {
    int offset = srcId * maxEdges;
    int freeSlot = -1;
    for (int i = 0; i < maxEdges; i++) {
      if (nodes[offset + i] == dstId) {
        return; // Already linked
      }
      if (nodes[offset + i] == -1 && freeSlot == -1) {
        freeSlot = i;
      }
    }

    if (freeSlot != -1) {
      nodes[offset + freeSlot] = dstId;
      edgeScores[offset + freeSlot] = score;
    } else {
      // Find weakest mathematically replacing implicitly physically
      // Preserve first 25% of edges chronologically to enforce Navigable Small World
      // long-range links!
      int evictStart = maxEdges / 4;
      int evictIdx = evictStart;
      float worstScore = edgeScores[offset + evictStart];
      for (int i = evictStart + 1; i < maxEdges; i++) {
        float s = edgeScores[offset + i];
        if (s < worstScore) {
          worstScore = s;
          evictIdx = i;
        }
      }
      if (score > worstScore) {
        nodes[offset + evictIdx] = dstId;
        edgeScores[offset + evictIdx] = score;
      }
    }
  }

  /**
   * Returns the contiguous list of neighboring document IDs. Values of -1 indicate empty capacity. As part of Graph
   * sorting metrics execution bounds phase, we automatically sort array bounds ascendingly.
   */
  public int[] getNeighbors(int docID) {
    if (docID * maxEdges < nodes.length) {
      int offset = docID * maxEdges;
      int[] out = new int[maxEdges];
      System.arraycopy(nodes, offset, out, 0, maxEdges);
      Arrays.sort(
          out); // Topological reordering of IDs conceptually minimizing localized memory jumps!
      return out;
    }
    return new int[0];
  }

  public int getSortedNeighbors(int docID, int[] destination) {
    if (docID * maxEdges < nodes.length) {
      int offset = docID * maxEdges;
      int validCount = 0;
      for (int i = 0; i < maxEdges; i++) {
        if (nodes[offset + i] != -1) {
          destination[validCount++] = nodes[offset + i];
        }
      }
      java.util.Arrays.sort(destination, 0, validCount);
      return validCount;
    }
    return 0;
  }

  /**
   * Tracks the total byte size consumed in RAM by the adjacency list and raw vector layers natively.
   */
  public long ramBytesUsed() {
    return (long) size * maxEdges * (Integer.BYTES + Float.BYTES);
  }

  /** Returns the count of distinct nodes indexed within the graph. */
  public int size() {
    return size;
  }
}
