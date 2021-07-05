package au.edu.deakin.eldt.utils;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;

import java.util.Random;

public class GenerateSyntheticPoint {

    private Type [] types = null;

    public double [] generate() {
        double [] vals = new double[types.length + 1]; // add the class

        for (int i = 0; i < types.length; i++) {
            Type type = types[i];

            vals[i] = type.generate();
        }

        return vals;
    }

    public void init(Instances data, Random random) {
        types = new Type[data.numAttributes() - 1];

        for (int i = 0; i < types.length; i++) {
            Attribute attribute = data.attribute(i);
            AttributeStats stats = data.attributeStats(i);

            if (attribute.isNominal()) {
                types[i] = new Nominal(attribute.numValues(), random);
            } else if (attribute.isNumeric()) {
                types[i] = new Numeric(stats.numericStats.min, stats.numericStats.max, random);
            }
        }
    }

}
