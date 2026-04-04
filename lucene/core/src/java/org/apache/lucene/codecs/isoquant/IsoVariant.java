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

/** Allowed variants for IsoQuant quantization. */
public enum IsoVariant {
  /** IsoQuant Full: v -> q_L v conjugate(q_R) */
  FULL(0),
  /** IsoQuant Fast: v -> q_L v */
  FAST(1),
  /** IsoQuant 2D: 2D Planar rotations mapped to 4D blocks */
  PLANAR_2D(2);

  public final int wireNumber;

  IsoVariant(int wireNumber) {
    this.wireNumber = wireNumber;
  }

  public static IsoVariant fromWireNumber(int wireNumber) {
    for (IsoVariant variant : values()) {
      if (variant.wireNumber == wireNumber) {
        return variant;
      }
    }
    throw new IllegalArgumentException("Unknown wireNumber for IsoVariant: " + wireNumber);
  }
}
