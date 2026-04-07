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
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.codecs.lsmvec.LsmVecGraph;
import org.apache.lucene.codecs.lsmvec.LsmVecGraphProvider;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
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
public class LsmVecWriterBenchmark {

  private LsmVecGraph graph;
  private Directory dir;
  private FieldInfo fieldInfo;
  
  @Param({"32"})
  int maxEdges;

  @Param({"100"})
  int efConstruction;
  
  @Param({"128"})
  int vectorDimension;

  @Param({"30000"})
  int numVectors;

  @Setup(Level.Trial)
  public void init() throws IOException {
    dir = new ByteBuffersDirectory();
    fieldInfo = new FieldInfo(
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
    graph = LsmVecGraphProvider.getInstance().getGraph(maxEdges, efConstruction, fieldInfo, vectorDimension, vectorsPerBlock);
    
    for (int i = 0; i < numVectors; i++) {
        graph.setVectorValue(i, floatVectors[i]);
        graph.addNode(i, i);
    }
  }

  @Benchmark
  public void flushGraphBytes() throws IOException {
      IndexOutput vectorDataOutput = dir.createOutput("test.vecd", IOContext.DEFAULT);
      IndexOutput vectorEdgeOutput = dir.createOutput("test.vem", IOContext.DEFAULT);
      
      int nodeBytes = Integer.highestOneBit((Integer.BYTES + maxEdges * Integer.BYTES) - 1) << 1;
      byte[] paddingBytesArr = new byte[nodeBytes];
      java.util.Arrays.fill(paddingBytesArr, (byte) 0xFF);
      int[] neighborBuffer = new int[maxEdges];
      
      for (int ord = 0; ord < numVectors; ord++) {
          int validEdges = graph.getSortedNeighbors(ord, neighborBuffer);
          vectorEdgeOutput.writeInt(validEdges);
          for (int i = 0; i < validEdges; i++) {
              vectorEdgeOutput.writeInt(neighborBuffer[i]);
          }
          int paddingBytes = nodeBytes - (Integer.BYTES + validEdges * Integer.BYTES);
          if (paddingBytes > 0) {
              vectorEdgeOutput.writeBytes(paddingBytesArr, 0, paddingBytes);
          }
      }
      
      vectorDataOutput.close();
      vectorEdgeOutput.close();
      dir.deleteFile("test.vecd");
      dir.deleteFile("test.vem");
  }
}
