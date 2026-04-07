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
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
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
public class LsmVecGraphBenchmark {

  private float[][] floatVectors;
  private FieldInfo fieldInfo;

  @Param({"32"})
  int maxEdges;

  @Param({"100"})
  int efConstruction;

  @Param({"128"})
  int vectorDimension;

  @Param({"10000"})
  int numVectors;

  @Setup(Level.Trial)
  public void init() {
    ThreadLocalRandom random = ThreadLocalRandom.current();
    floatVectors = new float[numVectors][vectorDimension];
    for (int i = 0; i < numVectors; ++i) {
      for (int j = 0; j < vectorDimension; j++) {
        floatVectors[i][j] = random.nextFloat();
      }
    }

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
  }

  @Benchmark
  public void buildLsmVecGraph() throws IOException {
    int vectorsPerBlock = Math.max(1, (1024 * 1024) / (vectorDimension * Float.BYTES));
    LsmVecGraph graph = LsmVecGraphProvider.getInstance().getGraph(maxEdges, efConstruction, fieldInfo, vectorDimension, vectorsPerBlock);

    for (int i = 0; i < numVectors; i++) {
      graph.setVectorValue(i, floatVectors[i]);
      graph.addNode(i, i);
    }
  }
}
