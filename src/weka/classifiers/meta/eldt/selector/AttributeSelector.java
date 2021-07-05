package weka.classifiers.meta.eldt.selector;

import weka.classifiers.meta.eldt.ranking.DefaultOrder;
import weka.classifiers.meta.eldt.ranking.Ranking;
import weka.core.Instances;
import weka.core.WekaException;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public abstract class AttributeSelector {

    private final Ranking m_Ranking = new DefaultOrder();

    private final ArrayList<Integer> m_Levels = new ArrayList<>();
    private final Set<Integer>       m_Set    = new HashSet<>();

    public void addToSet(int index) {
        m_Set.add(index);
    }

    public void init(Instances data, int minNumInst) throws Exception {
        m_Ranking.init(data, minNumInst);
    }

    public boolean removeFromSet(int index) {
        return m_Set.remove(index);
    }

    public int select(int level) throws Exception {
        int result = getSelect(level);

        if (result == -1) {
            return -1;
        } else {
            return m_Ranking.select(result);
        }
    }

    protected void addElement(int value) {
        m_Levels.add(value);
    }

    protected void reset() {
        m_Levels.clear();
        m_Set.clear();
    }

    private int getSelect(int level) throws WekaException {
        if (m_Levels.size() > level) {
            return m_Levels.get(level);
        } else {
            if (m_Set.isEmpty()) {
                return -1;
            }

            throw new WekaException("Should not be here anymore!");
        }
    }

}
