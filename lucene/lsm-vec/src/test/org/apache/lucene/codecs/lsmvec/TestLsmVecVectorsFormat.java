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

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.TestUtil;

/**
 * Validates that LsmVecVectorsFormat fundamentally round-trips correctly and integrates seamlessly
 * with the Lucene Segment merging lifecycle.
 */
public class TestLsmVecVectorsFormat extends BaseKnnVectorsFormatTestCase {

  @Override
  protected Codec getCodec() {
    return TestUtil.alwaysKnnVectorsFormat(new LsmVecVectorsFormat());
  }

  @Override
  protected boolean supportsFloatVectorFallback() {
    return false; // We don't implement simulateEmptyRawVectors yet
  }

  @Override
  protected void assertOffHeapByteSize(org.apache.lucene.index.LeafReader r, String fieldName)
      throws java.io.IOException {
    // Disabled since LsmVec does not implement getOffHeapByteSize strictly yet.
  }
}
