package weka.classifiers.meta.eldt.evaluator;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instances;

public abstract class NumericEvaluator extends Evaluator {

    abstract public void handleNumericAttribute(Distribution distribution, Instances data, int attributeIndex) throws Exception;

}
