package weka.classifiers.meta.eldt;

import au.edu.deakin.eldt.utils.GenerateSyntheticPoint;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.eldt.evaluator.NominalEvaluator;
import weka.classifiers.meta.eldt.evaluator.NumericEvaluator;
import weka.classifiers.meta.eldt.evaluator.RandomNominalEvaluator;
import weka.classifiers.meta.eldt.evaluator.RandomNumericEvaluator;
import weka.classifiers.meta.eldt.selector.AttributeSelector;
import weka.classifiers.trees.j48.Distribution;
import weka.core.*;

import java.util.ArrayList;
import java.util.Random;

public class Node {

    static final byte FLG_LEAF          = 0x01;
    static final byte FLG_LEVEL_TREE    = 0x02;
    static final byte FLG_NUMERIC       = 0x04;
    static final byte FLG_PURE          = 0x08;

    protected byte m_Flags = 0x00;

    protected NominalEvaluator m_NominalEvaluator  = null;
    protected NumericEvaluator m_NumericEvaluator  = null;

    protected Node[]       m_Children      = null;
    protected Distribution m_Distribution  = null;

    protected int m_AttributeIndex = 0;

    private final GenerateSyntheticPoint m_GenerateSyntheticPoint;

    private Classifier m_Classifier = null;

    public Node(GenerateSyntheticPoint generateSyntheticPoint) {
        m_GenerateSyntheticPoint = generateSyntheticPoint;
    }

    public void buildTree(Instances data, Classifier classifier, int minPts, AttributeSelector selector, int level, Random random) throws Exception {
        m_Flags = ((byte)0x00);
        m_NominalEvaluator = new RandomNominalEvaluator();
        m_NumericEvaluator = new RandomNumericEvaluator();

        m_NominalEvaluator.setRandom(random);
        m_NumericEvaluator.setRandom(random);

        m_Flags &= ~(FLG_LEAF | FLG_PURE);
        m_Flags |= FLG_LEVEL_TREE;

        m_Distribution = new Distribution(data);

    // Check if we have enough positive samples
        if (Utils.sm(m_Distribution.perClass(0), minPts)) {
            m_Flags |= FLG_LEAF;
            return;
        }

    // Check if we have a pure negative node
        if (Utils.eq(m_Distribution.total(), m_Distribution.perClass(1))) {
            m_Flags |= (FLG_LEAF | FLG_PURE);
            return;
        }

    // Pure positive node?
        if (Utils.eq(m_Distribution.total(), m_Distribution.perClass(0))) {
            buildClassifier(data, classifier);
            return;
        }

        int startLevel = level;
        ArrayList<Integer> attributeIndexes = new ArrayList<Integer>(data.numAttributes());

        for (;;) {
            m_AttributeIndex = selector.select(level);

            if (m_AttributeIndex == -1) {
                m_Distribution = new Distribution(data);
                m_AttributeIndex = selector.select(startLevel);
                m_Flags |= FLG_LEAF;

                for (Integer value : attributeIndexes) {
                    selector.addToSet(value);
                }

                buildClassifier(data, classifier);
                return;
            }

            if (!singleBranchTest(data)) {
                break;
            }

            if (!data.attribute(m_AttributeIndex).isNominal()) {
                if (selector.removeFromSet(m_AttributeIndex)) {
                    attributeIndexes.add(m_AttributeIndex);
                }
            }

            level++;
        }

        if (data.attribute(m_AttributeIndex).isNominal()) {
            m_Flags &= ~FLG_NUMERIC;
            m_NominalEvaluator.handleNominalAttribute(m_Distribution, data, m_AttributeIndex);
            m_Distribution = m_NominalEvaluator.getDistribution();
        } else {
            m_Flags |= FLG_NUMERIC;

            m_NumericEvaluator.handleNumericAttribute(m_Distribution, data, m_AttributeIndex);
            m_Distribution = m_NumericEvaluator.getDistribution();
        }

        Instances [] localInstances = split(data);

        m_Children = new Node[localInstances.length];

        for (int i = 0; i < localInstances.length; i++) {
            Node node = null;

            if (localInstances[i] != null) {
                node = new Node(m_GenerateSyntheticPoint);
                node.buildTree(localInstances[i], classifier, minPts, selector, level + 1, random);
            }

            m_Children[i] = node;
        }

        for (Integer attributeIndex : attributeIndexes) {
            selector.addToSet(attributeIndex);
        }
    }

    public double[] getProbs(Instance instance, double totalPositive) throws Exception {
        if ((m_Flags & FLG_LEVEL_TREE) == FLG_LEVEL_TREE) {
            if ((m_Flags & FLG_LEAF) == FLG_LEAF) {
                return classProb(instance, totalPositive);
            } else {
                int treeIndex = whichSubset(instance);

                if (m_Children[treeIndex] == null) {
                    return classProb(instance, totalPositive);
                } else {
                    return m_Children[treeIndex].getProbs(instance, totalPositive);
                }
            }
        } else {
            return m_Classifier.distributionForInstance(instance);
        }
    }

    private void buildClassifier(Instances data, Classifier classifier) throws Exception {
        Instances temp = new Instances(data, 0);
        int count = 0;

    // create a new set of synthetic points
        for (Instance instance : data) {
            if (instance.value(data.numAttributes() - 1) == 0.0) {
                count++;
                temp.add(instance);
            }
        }

        for (int i = 0; i < count; i++) {
            double [] vals = m_GenerateSyntheticPoint.generate();
            vals[vals.length - 1] = 1; // negative index

            Instance newInst = new DenseInstance(1.0, vals);
            temp.add(newInst);
        }

        m_Flags &= ~(FLG_LEAF | FLG_LEVEL_TREE | FLG_PURE);
        m_Classifier = AbstractClassifier.makeCopy(classifier);
        m_Classifier.buildClassifier(data);
    }

    private double[] classProb(Instance instance, double totalPositive) throws Exception {
        double[] probs = new double[instance.numClasses()];

        if ((m_Flags & FLG_PURE) == FLG_PURE) {
            probs[0] = 0.0;
            probs[1] = 1.0;
        } else {
            probs[0] = m_Distribution.perClass(0) / totalPositive;
            probs[1] = 1.0 - probs[0];
        }

        return probs;
    }

    private boolean singleBranchTest(Instances data) throws Exception {
        int missingCount = -1;
        double same = data.instance(0).value(m_AttributeIndex);
        int sameCount = -1;

        for (int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).isMissing(m_AttributeIndex)) {
                missingCount++;
            }

            if (Utils.eq(data.instance(i).value(m_AttributeIndex), same)) {
                sameCount++;
            }

            if ((missingCount != i) && (sameCount != i)) {
                return false;
            }
        }

        return true;
    }

    private Instances[] split(Instances data) throws Exception {
        Instances[] instances = new Instances[m_Distribution.numBags()];

        for (int i = 0; i < m_Distribution.numBags(); i++) {
            instances[i] = (m_Distribution.perBag(i) > 0) ? new Instances(data, (int)m_Distribution.perBag(i)) : null;
        }

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            int subset = whichSubset(instance);

            if (instances[subset] != null) {
                instances[subset].add(instance);
            } else {
                throw new WekaException("instances[" + subset + "] is null!");
            }
        }

        for (int i = 0; i < m_Distribution.numBags(); i++) {
            if (instances[i] != null) {
                instances[i].compactify();
            }
        }

        return instances;
    }

    private int whichSubset(Instance data) {
        if (data.attribute(m_AttributeIndex).isNominal()) {
            return m_NominalEvaluator.whichSubset(data, m_AttributeIndex);
        } else {
            return m_NumericEvaluator.whichSubset(data, m_AttributeIndex);
        }
    }

}
