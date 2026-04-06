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

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.Objects;
import org.apache.lucene.index.FieldInfo;

/**
 * A provider of LsmVecGraph implementations. Depending on the Java version and availability of native FFM modules, this
 * class provides either NativeLsmVecGraph or HeapLsmVecGraph.
 */
public abstract class LsmVecGraphProvider {

  /**
   * Returns the default instance of the provider matching native possibilities of actual runtime.
   */
  public static LsmVecGraphProvider getInstance() {
    return Objects.requireNonNull(Holder.INSTANCE, "call to getInstance() from subclass");
  }

  LsmVecGraphProvider() {
    // no instance/subclass except from this package
  }

  /** Returns an implementation of LsmVecGraph configured with the given parameters. */
  public abstract LsmVecGraph getGraph(
      int maxEdges,
      int efConstruction,
      FieldInfo fieldInfo,
      int vectorDimension,
      int vectorsPerBlock);

  // *** Lookup mechanism: ***

  static LsmVecGraphProvider lookup() {
    final String className = "Jdk25LsmVecGraphProvider";
    try {
      final var lookup = MethodHandles.lookup();
      final var cls = lookup.findClass("org.apache.lucene.codecs.lsmvec." + className);
      final var constr = lookup.findConstructor(cls, MethodType.methodType(void.class));
      return (LsmVecGraphProvider) constr.invoke();
    } catch (NoSuchMethodException | IllegalAccessException e) {
      throw new LinkageError(className + " is missing correctly typed constructor", e);
    } catch (ClassNotFoundException | UnsupportedOperationException _) {
      return new DefaultLsmVecGraphProvider();
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable th) {
      throw new AssertionError(th);
    }
  }

  /** This static holder class prevents classloading deadlock. */
  private static final class Holder {
    private Holder() {}

    static final LsmVecGraphProvider INSTANCE = lookup();
  }
}
