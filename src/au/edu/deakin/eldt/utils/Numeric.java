package au.edu.deakin.eldt.utils;

import java.util.Random;

public class Numeric extends Type {

    private final double m_Range;
    private final double m_Offset;

    public Numeric(double lower, double higher, Random random) {
        super(random);

        m_Offset = lower;
        m_Range = higher - lower;
    }

    public double generate() {
        return m_Random.nextDouble() * m_Range + m_Offset;
    }

}
