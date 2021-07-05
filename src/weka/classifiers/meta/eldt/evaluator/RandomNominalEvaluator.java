package weka.classifiers.meta.eldt.evaluator;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Enumeration;

public class RandomNominalEvaluator extends NominalEvaluator {

    protected int [] m_RandomAttributeIndex  = null;

    public void handleNominalAttribute(Distribution distribution,
                                       Instances data, int attributeIndex) throws Exception {
        int count = data.attribute(attributeIndex).numValues();

        m_RandomAttributeIndex = new int[count];
        ArrayList<Integer> randomAttributeIndexList = new ArrayList<Integer>(count);

        for(int s = 0; s < count; s++) {
            randomAttributeIndexList.add(s);
        }

        for(int s1 = 0; s1 < count; s1++) {
            int randomIndex = m_Random.nextInt(randomAttributeIndexList.size());
            Integer randNum = (Integer)randomAttributeIndexList.get(randomIndex);
            randomAttributeIndexList.remove(randomIndex);
            m_RandomAttributeIndex[s1] = randNum;
        }

        m_SplitPoint = 1;

        if (count > 2) {
            m_SplitPoint = m_Random.nextInt(count - 1) + 1;
        }

        m_Distribution = new Distribution(3, data.numClasses());

        Enumeration<Instance> enu = data.enumerateInstances();

        while (enu.hasMoreElements()) {
            Instance instance = enu.nextElement();
            int indexValue;

            if (instance.isMissing(attributeIndex)) {
                indexValue = 2;
            } else {
                indexValue = (int)instance.value(attributeIndex);

                for(int i = 0; i < m_RandomAttributeIndex.length; i++) {
                    if (indexValue == m_RandomAttributeIndex[i]) {
                        if (i < m_SplitPoint) {
                            indexValue = 0;
                        } else {
                            indexValue = 1;
                        }

                        break;
                    }
                }
            }

            m_Distribution.add(indexValue, instance);
        }
    }

    public int whichSubset(Instance data, int attributeIndex) {
        int indexValue = 0;

        if (data.isMissing(attributeIndex)) {
            indexValue = 2;
        } else {
            indexValue = (int)data.value(attributeIndex);

            for(int i = 0; i < m_RandomAttributeIndex.length; i++) {
                if (indexValue == m_RandomAttributeIndex[i]) {
                    if (i < m_SplitPoint) {
                        indexValue = 0;
                    } else {
                        indexValue = 1;
                    }

                    break;
                }
            }
        }

        return indexValue;
    }

}
