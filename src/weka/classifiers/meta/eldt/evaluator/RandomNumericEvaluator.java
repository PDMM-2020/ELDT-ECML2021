package weka.classifiers.meta.eldt.evaluator;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

public class RandomNumericEvaluator extends NumericEvaluator {

    @Override
    public void handleNumericAttribute(Distribution distribution, Instances data, int attributeIndex) throws Exception {
        int found = 0, indexInst = 0;

        for (int i = 0; (i < data.numInstances()) && (found < 2); i++) {
            if (!data.instance(i).isMissing(attributeIndex)) {
                found++;
                indexInst = i;
            }
        }

        double same = 0.0;
        int valueCount = -1;
        int sameCount = -1;

        for (int i = 0; i < data.numInstances(); i++) {
            if (!data.instance(i).isMissing(attributeIndex)) {
                same = data.instance(i).value(attributeIndex);
                break;
            }
        }

        boolean sameFound = true;

        for (int i = 0; i < data.numInstances(); i++) {
            if (!data.instance(i).isMissing(attributeIndex)) {
                valueCount++;
            }

            if (Utils.eq(data.instance(i).value(attributeIndex), same)) {
                sameCount++;
            }

            if (valueCount != sameCount) {
                sameFound =  false;
                break;
            }
        }

        if (sameFound) {
            m_SplitPoint = same;
        } else {
            switch (found) {
                case 0: // value for all instances are missing so it meaningless
                // Note: This case should now never occurs
                    m_SplitPoint = 0.0;
                    break;

                case 1: // found only only instances with a value
                    m_SplitPoint = data.instance(indexInst).value(attributeIndex);
                    break;

                default:        // have more than 1 instances with value
                    int first = m_Random.nextInt(data.numInstances());

                    while (data.instance(first).isMissing(attributeIndex)) {
                        first = m_Random.nextInt(data.numInstances());
                    }

                    int last = first;

                    double minValue = data.instance(first).value(attributeIndex);
                    double maxValue = data.instance(last).value(attributeIndex);

                    while (Utils.eq(minValue, maxValue)) {
                        last = m_Random.nextInt(data.numInstances());

                        while (data.instance(last).isMissing(attributeIndex)) {
                            last = m_Random.nextInt(data.numInstances());
                        }

                        maxValue = data.instance(last).value(attributeIndex);
                    }

                    m_SplitPoint = (minValue + maxValue) / 2.0;
                    break;
            }
        }

    //Split them up!
        Enumeration<Instance> enu = data.enumerateInstances();

        m_Distribution = new Distribution(3, data.numClasses());

        while (enu.hasMoreElements()) {
            Instance instance = enu.nextElement();

            if (!instance.isMissing(attributeIndex)) {
                if (Utils.smOrEq(instance.value(attributeIndex), m_SplitPoint)) {
                    m_Distribution.add(0, instance);
                } else {
                    m_Distribution.add(1, instance);
                }
            } else {
                m_Distribution.add(2, instance);
            }
        }
    }

    @Override
    public int whichSubset(Instance data, int attributeIndex) {
        if (data.isMissing(attributeIndex)) {
            return 2;
        } else {
            if (Utils.smOrEq(data.value(attributeIndex), m_SplitPoint)) {
                return 0;
            } else {
                return 1;
            }
        }
    }

}
