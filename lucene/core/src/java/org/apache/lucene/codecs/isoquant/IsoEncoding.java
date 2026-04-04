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

/** Allowed encodings for IsoQuant quantization. */
public enum IsoEncoding {
  /** 4-bit quantization, packed two values per byte. */
  FOUR_BIT(4, 0),
  /** 8-bit quantization, one value per byte. */
  EIGHT_BIT(8, 1);

  public final int bits;
  public final int wireNumber;

  IsoEncoding(int bits, int wireNumber) {
    this.bits = bits;
    this.wireNumber = wireNumber;
  }

  public static IsoEncoding fromWireNumber(int wireNumber) {
    for (IsoEncoding encoding : values()) {
      if (encoding.wireNumber == wireNumber) {
        return encoding;
      }
    }
    throw new IllegalArgumentException("Unknown wireNumber for IsoEncoding: " + wireNumber);
  }
}
