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

package org.apache.lucene.util;

import java.lang.foreign.MemorySegment;
import org.apache.lucene.internal.vectorization.VectorUtilSupport;
import org.apache.lucene.internal.vectorization.VectorUtilSupportFFM;
import org.apache.lucene.internal.vectorization.VectorizationProvider;

/**
 * Public facade to expose FFM-based operations leveraging MemorySegment for optimized vector processing.
 *
 * <p>This utility utilizes standard FFM features available in JDK 25 and delegating internally to
 * appropriate optimized CPU implementations (like native binaries or Panama incubator equivalents).
 */
public final class VectorUtilFFM {

  private static final VectorUtilSupportFFM IMPL;

  static {
    VectorUtilSupport support = VectorizationProvider.getInstance().getVectorUtilSupport();
    if (support instanceof VectorUtilSupportFFM) {
      IMPL = (VectorUtilSupportFFM) support;
    } else {
      IMPL = null;
    }
  }

  private VectorUtilFFM() {}

  /**
   * Returns true if FFM-based vector utilities are officially supported and loaded without fallback in this environment.
   */
  public static boolean isSupported() {
    return IMPL != null;
  }

  /**
   * Calculates the dot product of the given float MemorySegment vectors.
   */
  public static float dotProductFloat(MemorySegment a, MemorySegment b) {
    if (IMPL == null) {
        throw new UnsupportedOperationException("FFM vectorization not supported on this runtime");
    }
    return IMPL.dotProductFloat(a, b);
  }

  /**
   * Calculates the cosine similarity of the given float MemorySegment vectors.
   */
  public static float cosineFloat(MemorySegment a, MemorySegment b) {
    if (IMPL == null) {
        throw new UnsupportedOperationException("FFM vectorization not supported on this runtime");
    }
    return IMPL.cosineFloat(a, b);
  }

  /**
   * Calculates the square distance of the given float MemorySegment vectors.
   */
  public static float squareDistanceFloat(MemorySegment a, MemorySegment b) {
    if (IMPL == null) {
        throw new UnsupportedOperationException("FFM vectorization not supported on this runtime");
    }
    return IMPL.squareDistanceFloat(a, b);
  }
}
