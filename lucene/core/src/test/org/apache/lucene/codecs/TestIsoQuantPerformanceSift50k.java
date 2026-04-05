package org.apache.lucene.codecs;

import java.io.InputStream;
import org.junit.Assume;

public class TestIsoQuantPerformanceSift50k extends BaseIsoQuantPerformanceTest {

  @Override
  public void setUp() throws Exception {
    Assume.assumeTrue(
        "SIFT50k dataset is not available in resources",
        TestIsoQuantPerformanceSift50k.class.getResource("sift50k_sampled/sift_base.fvecs") != null);
    super.setUp();
  }

  @Override
  protected InputStream getBaseInputStream() throws Exception {
    return TestIsoQuantPerformanceSift50k.class.getResourceAsStream("sift50k_sampled/sift_base.fvecs");
  }

  @Override
  protected InputStream getQueryInputStream() throws Exception {
    return TestIsoQuantPerformanceSift50k.class.getResourceAsStream("sift50k_sampled/sift_query.fvecs");
  }

  @Override
  protected InputStream getGroundtruthInputStream() throws Exception {
    return TestIsoQuantPerformanceSift50k.class.getResourceAsStream("sift50k_sampled/sift_groundtruth.ivecs");
  }

  @Override
  protected String getDatasetName() {
    return "SIFT50k";
  }

  @Override
  protected int getM() { return 64; }

  @Override
  protected int getEfConstruction() { return 400; }

  @Override
  protected int getEfSearch() { return 100; }
}
