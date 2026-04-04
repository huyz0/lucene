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
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * Native IsoQuant Flat Vectors Format. Pack 4-bit indices and norm directly.
 *
 * @lucene.experimental
 */
public final class IsoQuantFlatFormat extends FlatVectorsFormat {

  static final String NAME = "IsoQuantFlatFormat";
  static final String META_CODEC_NAME = "IsoQuantFlatFormatMeta";
  static final String VECTOR_DATA_CODEC_NAME = "IsoQuantFlatFormatData";
  static final String RAW_DATA_CODEC_NAME = "IsoQuantFlatFormatRawData";
  static final String META_EXTENSION = "iqm";
  static final String VECTOR_DATA_EXTENSION = "iqd";
  static final String RAW_DATA_EXTENSION = "iqr";

  public static final int VERSION_START = 0;
  public static final int VERSION_CURRENT = VERSION_START;

  static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;
  public static final int LZ4_BLOCK_SHIFT = 5;

  private final IsoEncoding encoding;
  private final IsoVariant variant;
  private final IsoCompress compress;
  private final long seed;

  /** Constructs a format */
  public IsoQuantFlatFormat(IsoEncoding encoding, IsoVariant variant, IsoCompress compress, long seed) {
    super(NAME);
    this.encoding = encoding;
    this.variant = variant;
    this.compress = compress;
    this.seed = seed;
  }

  @Override
  public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new IsoQuantFlatWriter(state, new IsoQuantFlatVectorScorer(seed), encoding, variant, compress, seed);
  }

  @Override
  public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new IsoQuantFlatReader(
        state,
        new IsoQuantFlatVectorScorer(seed),
        seed); // the reader will read bits directly from the meta file
  }

  @Override
  public String toString() {
    return "IsoQuantFlatFormat(seed=" + seed + ')';
  }
}
