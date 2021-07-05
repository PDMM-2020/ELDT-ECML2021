package weka.classifiers.meta.eldt.ranking;

import weka.core.Instances;

public class Ranking {

    private int [] m_Index = null;

    public void addAt(int idx, int value) {
        m_Index[idx] = value;
    }

    public void init(Instances data, int minNumInst) throws Exception {
        m_Index = new int[data.numAttributes() - 1];
    }

    public int select(int index) {
        return m_Index[index];
    }

}
