package weka.classifiers.meta;

import au.edu.deakin.eldt.utils.GenerateSyntheticPoint;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.meta.eldt.Node;
import weka.classifiers.meta.eldt.selector.Randomise;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.util.Random;

public class ELDT extends SingleClassifierEnhancer {

    private final Random                 m_Random;
    private final GenerateSyntheticPoint m_GenerateSyntheticPoint;

    private boolean m_UseHeight = false;
    private boolean m_UseMinPts = false;

    private int m_RequestedHeight    =   1;
    private int m_RequestedMinPts    =   2;
    private int m_RequestedTreeCount = 100;

    private int    m_MaxHeight          =    1;
    private int    m_MinNumInstances    =    2;
    private double m_TotalPositiveClass = -1.0;

    private Node [] m_Trees = null;

    public ELDT(Random random, GenerateSyntheticPoint generateSyntheticPoint) {
        m_Random = random;
        m_GenerateSyntheticPoint = generateSyntheticPoint;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
    // sanity test!
        getCapabilities().testWithFail(data);

        m_TotalPositiveClass = data.numInstances();

        if (m_UseMinPts) {
            m_MinNumInstances = m_RequestedMinPts;
        } else {
            m_MinNumInstances = (int) (Math.floor(Utils.log2(m_TotalPositiveClass)) + 1);
        }

        Randomise selector = new Randomise();

        if (m_UseHeight) {
            m_MaxHeight = m_RequestedHeight;
        } else {
            m_MaxHeight = (int) (Math.floor(Utils.log2(data.numAttributes())) + 1);
        }

        selector.setRequestedHeight(m_MaxHeight);
        selector.setRequestedTreeCount(m_RequestedTreeCount);
        selector.init(data, m_MinNumInstances, m_Random);
        m_Trees = new Node[m_RequestedTreeCount];

        for (int i = 0; i < m_RequestedTreeCount; i++) {
            selector.setTreeIndex(i);
            Instances temp = getTrainingSet(data);
            m_Trees[i] = new Node(m_GenerateSyntheticPoint);
            m_Trees[i].buildTree(temp, m_Classifier, m_MinNumInstances, selector, 0, m_Random);
        }
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        double [] probs = new double[instance.numClasses()];

        for (Node tree : m_Trees) {
            double [] prob = tree.getProbs(instance, m_TotalPositiveClass);

            for (int j = 0; j < prob.length; j++) {
                probs[j] += prob[j];
            }
        }

        for (int i = 0; i < probs.length; i++) {
            probs[i] /= m_Trees.length;
        }

        return probs;
    }

    private Instances getTrainingSet(Instances data) throws Exception {
        Instances newData = new Instances(data, 0);

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            newData.add(instance);
        }

    // now add the synthetic data points as the negative class
        for (int i = 0; i < data.numInstances(); i++) {
            double [] vals = m_GenerateSyntheticPoint.generate();
            vals[vals.length - 1] = 1; // negative index

            Instance newInst = new DenseInstance(1.0, vals);
            newData.add(newInst);
        }

    // last step is to randomise it
        Randomize randomize = new Randomize();
        randomize.setInputFormat(newData);
        randomize.setRandomSeed(42);
        newData = Filter.useFilter(newData, randomize);

        return newData;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        m_UseHeight = Utils.getFlag("useHeight", options);
        m_UseMinPts = Utils.getFlag("useMinPts", options);

        if (m_UseHeight) {
            String heightString = Utils.getOption("H", options);

            if (heightString.isEmpty()) {
                m_RequestedHeight = 1;
            } else {
                m_RequestedHeight = Integer.parseInt(heightString);
            }
        }

        if (m_UseMinPts) {
            String minPtsString = Utils.getOption("minPts", options);

            if (minPtsString.isEmpty()) {
                m_RequestedMinPts = 2;
            } else {
                m_RequestedMinPts = Integer.parseInt(minPtsString);
            }
        }

        String treeCountString = Utils.getOption("T", options);

        if (treeCountString.length() > 0) {
            m_RequestedTreeCount = Integer.parseInt(treeCountString);
        } else {
            m_RequestedTreeCount = 100;
        }

        super.setOptions(options);
    }

    public String toString() {
        if (m_Trees == null) {
            return "No models built!";
        }

        String result = "ELDT\n====";

        result += "\nBase Classifier: " + m_Classifier.getClass().getSimpleName();
        result += "\nUse Height     : " + m_UseHeight;
        result += "\nUse MinPts     : " + m_UseMinPts;
        result += "\nHeight         : " + m_MaxHeight;
        result += "\nNumber of trees: " + m_Trees.length;
        result += "\nMinimum number of Instances: " + m_MinNumInstances;

        return result;
    }

}
