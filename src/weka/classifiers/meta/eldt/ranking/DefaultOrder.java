package weka.classifiers.meta.eldt.ranking;

import weka.core.Instances;

public class DefaultOrder extends Ranking {

    public void init(Instances instances, int minNumInst) throws Exception {
        super.init(instances, minNumInst);

        for (int i = instances.numAttributes() - 2; i > -1; i--) {
            addAt(i, i);
        }
    }

}
