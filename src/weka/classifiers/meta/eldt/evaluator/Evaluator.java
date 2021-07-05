package weka.classifiers.meta.eldt.evaluator;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public abstract class Evaluator {

    protected Distribution  m_Distribution  = null;
    protected double        m_SplitPoint    = 0.0;

    protected Random m_Random;

    public Distribution getDistribution() {
        return m_Distribution;
    }

    public void setRandom(Random random) {
        m_Random = random;
    }

    abstract public int whichSubset(Instance data, int attributeIndex);

}
