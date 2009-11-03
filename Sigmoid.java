import java.lang.Math;

/// Sigmoid implemented using logistic function
public class Sigmoid implements ActivationFunction {
	public Sigmoid() {}
	
	//public Sigmoid(double beta) { this.beta = beta }
	
	public double activate(double x) {
		return 1/(1+Math.exp(-x));
	}
	
	public double derived_activate(double x) {
		return activate(x) * (1 - activate(x));
	}
}
