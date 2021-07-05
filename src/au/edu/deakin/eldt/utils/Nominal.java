package au.edu.deakin.eldt.utils;

import java.util.Random;

public class Nominal extends Type {

    private final int m_Count;

    public Nominal(int count, Random random) {
        super(random);
        m_Count = count;
    }

    public double generate() {
        return m_Random.nextInt(m_Count);
    }

}
