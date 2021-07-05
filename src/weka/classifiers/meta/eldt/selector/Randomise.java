package weka.classifiers.meta.eldt.selector;

import weka.core.Instances;

import java.util.*;

public class Randomise extends Height {

    private int m_RequestedTreeCount = 100;

    private int m_NumAttributes     = 0;
    private int m_OriginalTreeCount = 0;

    private Map<Integer, String> m_Map    = null;
    private Random               m_Random = null;

    public void init(Instances data, int minNumInst, Random random) throws Exception {
        super.init(data, minNumInst);

        m_NumAttributes = data.numAttributes() - 1;
        m_OriginalTreeCount = getTreeCount();

        setTreeCount(m_RequestedTreeCount);

        m_Map = new HashMap<Integer, String>();
        m_Random = random;
    }

    public void setRequestedTreeCount(int requestedTreeCount) {
        m_RequestedTreeCount = requestedTreeCount;
    }

    public void setTreeIndex(int index) {
        reset();

        int roundNum = index / m_OriginalTreeCount;

        if (m_Map.size() == getTreeCount()) {
            restoreFill(index);

            for (int i = 0; i < m_Height; i++) {
                addElement(m_Index[i]);
            }
        } else {
            StringBuilder treeValues = new StringBuilder();

            do {
                randomFill();
                treeValues = new StringBuilder();

                for (int i = 0; i < m_Height; i++) {
                    treeValues.append(m_Index[i]).append(",");
                }

                treeValues.append(roundNum);
            } while(m_Map.containsValue(treeValues.toString()));

            for (int i = 0; i < m_Height; i++) {
                addElement(m_Index[i]);
            }

            m_Map.put(index, treeValues.toString());
        }
    }

    private void randomFill() {
        int [] L = new int[m_Height];

        Arrays.fill(L, 1);

        for(int i = 0; i < L.length; i++) {
            if (i == 0) {
                L[i] = m_Random.nextInt(m_NumAttributes - m_Height + 1) + 1;
            }

            if (i > 0) {
                if ((m_NumAttributes - m_Height + (i + 1) - L[i - 1]) == 0) {
                    L[i] = L[i - 1] + 1;
                } else {
                    L[i] = m_Random.nextInt(m_NumAttributes - m_Height + (i + 1) - L[i - 1]) + L[i - 1] + 1;
                }
            }

            m_Index[i] = L[i] - 1; // rescale to start the index from 0 rather than 1
        }
    }

    private void restoreFill(int index) {
        String storedTreeValues = m_Map.get(index);
        StringTokenizer tokens;
        String Delimiters = ",";
        tokens = new StringTokenizer(storedTreeValues, Delimiters);

    // extract the data from the input string
        for (int i = 0; i < m_Index.length; i++) {
            m_Index[i] = Integer.parseInt(tokens.nextToken());
        }
    }

}
