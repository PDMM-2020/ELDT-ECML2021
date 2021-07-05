package au.edu.deakin.eldt.experiments;

import au.edu.deakin.eldt.utils.GenerateSyntheticPoint;
import weka.classifiers.meta.ELDT;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RunELDT {

    private Instances m_Train;
    private Instances m_Test;

    private final Map<Double, List<Integer>> m_MapTest = new TreeMap<>();

    private final GenerateSyntheticPoint m_GenerateSyntheticPoint = new GenerateSyntheticPoint();

    private Random m_Random;

    List<Double> m_Scores = new ArrayList<>();

    public static void main(String [] args) {
        try {
            RunELDT exp = new RunELDT();
            exp.go(args);
        } catch (Exception ex) {
            Logger.getLogger(RunELDT.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void go(String [] args) throws Exception {
        System.out.println("Running ELDT experiments with additional synthetic points.");

        m_Train = loadData("Train", args);
        m_Test = remapTestData(loadData("Test", args));

        init(args);

        doRun(args);

        System.out.println("AUC: " + computeAUC(m_Scores));
    }

    private double computeAUC(List<Double> list) {
        int [] idx = Utils.stableSort(list.stream().mapToDouble(d -> d).toArray()); //ascending order

        double clsAnomaly = 0.0;
        double min = Double.MAX_VALUE;

    // find the anomaly class label
        for (Map.Entry<Double, List<Integer>> set : m_MapTest.entrySet()) {
            if (set.getValue().size() < min) {
                min = set.getValue().size();
                clsAnomaly = set.getKey();
            }
        }

    // compute the AUC
        double tp = 0.0;
        double fp = 0.0;
        double sum = 0.0;

        for (int i : idx) {
            for (Map.Entry<Double, List<Integer>> set : m_MapTest.entrySet()) {
                if (set.getValue().contains(i)) {
                    if (set.getKey() == clsAnomaly) {
                        tp++;
                    } else {
                        fp++;
                        sum += tp;
                    }
                }
            }
        }

        return sum / (tp * fp);
    }

    private void doRun(String [] args) throws Exception {
        ELDT classifier = new ELDT(m_Random, m_GenerateSyntheticPoint);
        classifier.setOptions(args);

        Instances data = generateTrainingData();
        classifier.buildClassifier(data);
        System.out.println(classifier);
        System.out.println();

        int modCount = 0;
        int mod = m_Test.numInstances();
        mod = (mod > 100) ? mod / 100 : 1;

        for (Instance instance : m_Test) {
            modCount++;

            if ((modCount % mod) == 0) {
                System.out.print("T");
            }

            double [] dist = classifier.distributionForInstance(instance);

        // add the score of the positive class
            m_Scores.add(dist[0]);
        }

        System.out.println();
    }

    private Instances generateTrainingData() throws Exception {
        Instances newData = new Instances(m_Train, 0);

    // remove the real class details
        newData.setClassIndex(-1);
        newData.deleteAttributeAt(newData.numAttributes() - 1);

    // add a new class details
        List<String> nominalValues = Arrays.asList("+1", "-1");
        newData.insertAttributeAt(new Attribute("PosNeg", nominalValues), newData.numAttributes());
        newData.setClassIndex(newData.numAttributes() - 1);

    // now remap the training data to the positive class
        for (int i = 0; i < m_Train.numInstances(); i++) {
            Instance instance = m_Train.instance(i);
            double [] vals = new double[newData.numAttributes()];

            for (int j = 0; j < newData.numAttributes(); j++) {
                vals[j] = instance.value(j);
            }

            vals[vals.length - 1] = 0; // positive index

            Instance newInst = (instance instanceof SparseInstance) ? new SparseInstance(instance.weight(), vals) : new DenseInstance(instance.weight(), vals);
            newData.add(newInst);
        }

    // last step is to randomise it
        Randomize randomize = new Randomize();
        randomize.setInputFormat(newData);
        randomize.setRandomSeed(42);
        newData = Filter.useFilter(newData, randomize);

        return newData;
    }

    private void init(String [] args) throws Exception {
        String seed = Utils.getOption("randomSeed", args);

        if (seed.isEmpty()) {
            throw new Exception("No random seed given.");
        }

        m_Random = new Random(Integer.parseInt(seed));
        m_GenerateSyntheticPoint.init(m_Train, m_Random);
    }

    private Instances loadData(String key, String [] args) throws Exception {
        String fileName = Utils.getOption("data" + key + "File", args);

        if (fileName.isEmpty()) {
            throw new Exception("No data " + key + " file name given.");
        }

        DataSource source = new DataSource(fileName);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        return data;
    }

    private Instances remapTestData(Instances data) {
        Instances newData = new Instances(data, 0);

    // remove the real class details
        newData.setClassIndex(-1);
        newData.deleteAttributeAt(newData.numAttributes() - 1);

    // add a new class details
        List<String> nominalValues = Arrays.asList("+1", "-1");
        newData.insertAttributeAt(new Attribute("PosNeg", nominalValues), newData.numAttributes());
        newData.setClassIndex(newData.numAttributes() - 1);

    // now remap the test data to positive class
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);

            double [] vals = new double[newData.numAttributes()];

            for (int j = 0; j < newData.numAttributes(); j++) {
                vals[j] = instance.value(j);
            }

            vals[vals.length - 1] = 0; // positive index

            Instance newInst = (instance instanceof SparseInstance) ? new SparseInstance(instance.weight(), vals) : new DenseInstance(instance.weight(), vals);
            newData.add(newInst);

            m_MapTest.computeIfAbsent(instance.classValue(), k -> new ArrayList<>()).add(i);
        }

        return newData;
    }

}
