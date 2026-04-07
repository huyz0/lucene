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
import java.nio.file.Files;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 4, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xmx4g", "-Xms4g", "-XX:+AlwaysPreTouch", "--add-modules=jdk.incubator.vector"})
public class LsmVecReaderBenchmark {
  private IndexInput edgeInput;
  private int[] entryNeighbors;
  private long nodeBytes;
  private int[] fetchBuffer;

  @Param({"32"})
  int maxEdges;

  @Param({"100"})
  int efConstruction;

  @Param({"128"})
  int vectorDimension;

  @Param({"10000"})
  int numVectors;

  @Setup(Level.Trial)
  public void init() throws IOException {
    Directory dir = new MMapDirectory(Files.createTempDirectory("lsmvec_bench_"));
    FieldInfo fieldInfo = new FieldInfo(
        "vector_field",
        1,
        false, false, false,
        IndexOptions.NONE,
        org.apache.lucene.index.DocValuesType.NONE,
        org.apache.lucene.index.DocValuesSkipIndexType.NONE,
        -1L,
        java.util.Collections.emptyMap(),
        0, 0, 0,
        vectorDimension,
        VectorEncoding.FLOAT32,
        VectorSimilarityFunction.DOT_PRODUCT,
        false, false);

    ThreadLocalRandom random = ThreadLocalRandom.current();
    float[][] floatVectors = new float[numVectors][vectorDimension];
    for (int i = 0; i < numVectors; ++i) {
      for (int j = 0; j < vectorDimension; j++) {
        floatVectors[i][j] = random.nextFloat();
      }
    }

    int vectorsPerBlock = Math.max(1, (1024 * 1024) / (vectorDimension * Float.BYTES));
    LsmVecGraph graph = LsmVecGraphProvider.getInstance().getGraph(maxEdges, efConstruction,
        fieldInfo, vectorDimension, vectorsPerBlock);

    for (int i = 0; i < numVectors; i++) {
        graph.setVectorValue(i, floatVectors[i]);
        graph.addNode(i, i);
    }

    IndexOutput vectorEdgeOutput = dir.createOutput("test.vem", IOContext.DEFAULT);

    nodeBytes = Integer.highestOneBit((Integer.BYTES + maxEdges * Integer.BYTES) - 1) << 1;
    byte[] paddingBytesArr = new byte[(int) nodeBytes];
    java.util.Arrays.fill(paddingBytesArr, (byte) 0xFF);
    int[] neighborBuffer = new int[maxEdges];

    for (int ord = 0; ord < numVectors; ord++) {
        int validEdges = graph.getSortedNeighbors(ord, neighborBuffer);
        vectorEdgeOutput.writeInt(validEdges);
        for (int i = 0; i < validEdges; i++) {
            vectorEdgeOutput.writeInt(neighborBuffer[i]);
        }
        int paddingBytes = (int) nodeBytes - (Integer.BYTES + validEdges * Integer.BYTES);
        if (paddingBytes > 0) {
            vectorEdgeOutput.writeBytes(paddingBytesArr, 0, paddingBytes);
        }
    }
    vectorEdgeOutput.close();

    edgeInput = dir.openInput("test.vem", IOContext.DEFAULT);

    edgeInput.seek(0);
    int entryNumNeighbors = edgeInput.readInt();
    entryNeighbors = new int[entryNumNeighbors];
    for (int i = 0; i < entryNumNeighbors; i++) {
        entryNeighbors[i] = edgeInput.readInt();
    }
    fetchBuffer = new int[maxEdges];
  }

  @Benchmark
  public void searchGraphTop10() throws IOException {
      int topK = 10;
      NeighborQueue candidates = new NeighborQueue(topK, false);
      NeighborQueue results = new NeighborQueue(topK, false);
      BitSet visited = new FixedBitSet(numVectors);

      int entryDoc = 0;
      float minAcceptedSimilarity = Float.NEGATIVE_INFINITY;

      candidates.add(entryDoc, 1.0f);
      results.insertWithOverflow(entryDoc, 1.0f);
      visited.set(entryDoc);

      while (candidates.size() > 0) {
        float currentScore = candidates.topScore();
        int topNodeOrd = candidates.pop();

        if (currentScore < minAcceptedSimilarity) {
          break;
        }

        if (topNodeOrd == 0 && entryNeighbors != null) {
          int numNeighbors = entryNeighbors.length;
          for (int i = 0; i < numNeighbors; i++) {
            int nDoc = entryNeighbors[i];
            if (!visited.getAndSet(nDoc)) {
              float s = 0.5f; // mock scorer
              if (s >= minAcceptedSimilarity) {
                candidates.add(nDoc, s);
                if (results.insertWithOverflow(nDoc, s)) {
                }
              }
            }
          }
        } else {
          edgeInput.seek((long) topNodeOrd * nodeBytes);
          int numNeighbors = edgeInput.readInt();
          edgeInput.readInts(fetchBuffer, 0, numNeighbors);
          for (int i = 0; i < numNeighbors; i++) {
            int nDoc = fetchBuffer[i];
            if (!visited.getAndSet(nDoc)) {
              float s = 0.5f; // mock scorer
              if (s >= minAcceptedSimilarity) {
                candidates.add(nDoc, s);
                if (results.insertWithOverflow(nDoc, s)) {
                }
              }
            }
          }
        }
      }
  }
}
