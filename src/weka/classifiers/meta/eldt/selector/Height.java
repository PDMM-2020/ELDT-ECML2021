package weka.classifiers.meta.eldt.selector;

import weka.core.Instances;

public class Height extends AttributeSelector {

    protected int    m_Height = 1;
    protected int [] m_Index  = null;

    private int m_RequestedHeight   = 1;

    private int m_TreeCount         = 0;

    public int getTreeCount() {
        return m_TreeCount;
    }

    public void init(Instances data, int minNumInst) throws Exception {
        super.init(data, minNumInst);

        int actualTotalHeight = data.numAttributes() - 1;
        m_Height = actualTotalHeight / 2;

        if ((actualTotalHeight % 2) != 0) {
            m_Height++;
        }

        if (m_RequestedHeight < m_Height) {
            m_Height = m_RequestedHeight;
        }

        int totalTrees = 1;

        for (int i = 1, j = actualTotalHeight; i <= m_Height; i++, j--) {
            totalTrees = totalTrees / i * j + totalTrees % i * j / i; // totalTrees * j / i;
        }

        setTreeCount(totalTrees);

        m_Index = new int[m_Height];
    }

    public void setRequestedHeight(int requestedHeight) {
        m_RequestedHeight = requestedHeight;
    }

    protected void setTreeCount(int treeCount) {
        m_TreeCount = treeCount;
    }

}
