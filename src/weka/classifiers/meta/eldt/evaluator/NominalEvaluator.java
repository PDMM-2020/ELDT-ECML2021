package weka.classifiers.meta.eldt.evaluator;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

public class NominalEvaluator extends Evaluator {

    public void handleNominalAttribute(Distribution distribution, Instances data, int attributeIndex) throws Exception {
        int count = data.attribute(attributeIndex).numValues();

        m_Distribution = new Distribution(count + 1, data.numClasses());

        Enumeration<Instance> enu = data.enumerateInstances();

        while (enu.hasMoreElements()) {
            Instance instance = enu.nextElement();

            int indexValue;

            if (instance.isMissing(attributeIndex)) {
                indexValue = count;
            } else {
                indexValue = (int)instance.value(attributeIndex);
            }

            m_Distribution.add(indexValue, instance);
        }
    }

    public int whichSubset(Instance data, int attributeIndex) {
        if (data.isMissing(attributeIndex)) {
            return data.attribute(attributeIndex).numValues();
        } else {
            return (int)data.value(attributeIndex);
        }
    }

}
