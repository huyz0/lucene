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

import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;

/**
 * FilterCodec that delegates everything to the default Lucene 104 codec, except for {@link KnnVectorsFormat}, which is
 * replaced with the native Log-Structured Merge-tree based Vector Exact Clustering graph algorithm
 * ({@link LsmVecVectorsFormat}).
 */
public final class LsmVecCodec extends FilterCodec {

  private final KnnVectorsFormat knnVectorsFormat;

  /** Instantiates the LsmVecCodec with the default window and max edges graph parameters. */
  public LsmVecCodec() {
    this(new LsmVecVectorsFormat());
  }

  /**
   * Instantiates the LsmVecCodec wrapping a custom initialized LsmVecVectorsFormat.
   *
   * @param lsmVecVectorsFormat the custom KnnVectorsFormat
   */
  public LsmVecCodec(KnnVectorsFormat lsmVecVectorsFormat) {
    super("LsmVec", new Lucene104Codec());
    this.knnVectorsFormat = lsmVecVectorsFormat;
  }

  @Override
  public KnnVectorsFormat knnVectorsFormat() {
    return knnVectorsFormat;
  }
}
