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
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * End-to-end IsoQuant Codec bringing HNSW and Native 4-bit Lloyd-Max
 * pre-quantized Flat Vectors
 * together.
 */
public final class IsoQuantHnswVectorsFormat extends KnnVectorsFormat {

  static final String NAME = "IsoQuantHnswVectorsFormat";
  private final int maxConn;
  private final int beamWidth;
  private final FlatVectorsFormat flatVectorsFormat;

  /** No-arg constructor for SPI compatibility. */
  public IsoQuantHnswVectorsFormat() {
    this(IsoEncoding.FOUR_BIT, IsoVariant.FAST, IsoCompress.NONE, 16, 100, 12345L);
  }

  public IsoQuantHnswVectorsFormat(int maxConn, int beamWidth, IsoEncoding encoding, IsoVariant variant, IsoCompress compress, long seed) {
    this(encoding, variant, compress, maxConn, beamWidth, seed);
  }

  public IsoQuantHnswVectorsFormat(IsoEncoding encoding, IsoVariant variant, IsoCompress compress, int maxConn, int beamWidth, long seed) {
    super(NAME);
    this.maxConn = maxConn;
    this.beamWidth = beamWidth;
    this.flatVectorsFormat = new IsoQuantFlatFormat(encoding, variant, compress, seed);
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    FlatVectorsWriter flatWriter = flatVectorsFormat.fieldsWriter(state);
    return new Lucene99HnswVectorsWriter(state, maxConn, beamWidth, flatWriter, 1, null);
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    FlatVectorsReader flatReader = flatVectorsFormat.fieldsReader(state);
    return new Lucene99HnswVectorsReader(state, flatReader);
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return 1024;
  }

  @Override
  public String toString() {
    return "IsoQuantHnswVectorsFormat(maxConn="
        + maxConn
        + ", beamWidth="
        + beamWidth
        + ", flatFormat="
        + flatVectorsFormat
        + ")";
  }
}
