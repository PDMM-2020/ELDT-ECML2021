package au.edu.deakin.eldt.utils;

import java.util.Random;

public abstract class Type {

    protected Random m_Random;

    public Type(Random random) {
        m_Random = random;
    }

    public abstract double generate();

}
