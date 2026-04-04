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

/** Compression level for raw 32-bit floats in IsoQuant index */
public enum IsoCompress {
  NONE((byte) 0),
  LZ4((byte) 1),
  ZSTD((byte) 2),
  LZAV((byte) 3);

  public final byte wireNumber;

  IsoCompress(byte wireNumber) {
    this.wireNumber = wireNumber;
  }

  public static IsoCompress fromWireNumber(int wireNumber) {
    for (IsoCompress c : values()) {
      if (c.wireNumber == wireNumber) {
        return c;
      }
    }
    throw new IllegalArgumentException("Unknown wire number for IsoCompress: " + wireNumber);
  }
}
