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
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * Native LSM-VEC Vector format.
 *
 * <p>Encodes vectors and implements the Log-Structured Merge-tree based Vector Exact Clustering
 * graph algorithm strictly storing nodes sequentially. It delegates to the {@link Lucene99FlatVectorsFormat} to
 * serialize the raw vector payloads into `.vec` formats, while strictly managing the structural graph into `.vem`
 * (Vector Edge Metadata) via native append-only streams.
 */
public final class LsmVecVectorsFormat extends KnnVectorsFormat {

  public static final String NAME = "LsmVec";

  // Extension for the Vector Edge Metadata graph
  static final String VEM_EXTENSION = "vem";
  static final String VEM_META_EXTENSION = "vma";

  // Extension for the pristine flat vector data directly
  static final String VECD_EXTENSION = "vecd";

  public static final int DEFAULT_MAX_EDGES = 32;
  public static final int DEFAULT_EF_CONSTRUCTION = 100;

  private final int maxEdges;
  private final int efConstruction;

  public LsmVecVectorsFormat() {
    this(DEFAULT_MAX_EDGES, DEFAULT_EF_CONSTRUCTION);
  }

  public LsmVecVectorsFormat(int maxEdges, int efConstruction) {
    super(NAME);
    this.maxEdges = maxEdges;
    this.efConstruction = efConstruction;
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new LsmVecVectorsWriter(state, maxEdges, efConstruction);
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new LsmVecVectorsReader(state);
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return 4096;
  }
}
