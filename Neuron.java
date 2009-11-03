import java.util.Random;

public final class Neuron {
	protected double[] weights;
	protected double bias;
	protected ActivationFunction act_func;
	protected double cached_netsum;
	
    public Neuron(int number_of_inputs, double bias, ActivationFunction f) {
        weights = new double[number_of_inputs];
		this.bias = bias;
		act_func = f;
		
		// Randomly initialize weights:
		Random r = new Random();
		for (int i = 0; i < weights.length; i++)
			weights[i] = r.nextDouble(); // 0-1
    }

	public double netsum(double[] inputs) {
		double netsum = bias;

		for (int i = 0; i < inputs.length; i++)
			netsum += inputs[i] * weights[i];
			
		// cache it, for use with backprop later
		cached_netsum = netsum;
			
		return netsum;
	}
	
	public double activate(double netsum) {
		//System.out.println("activate(" + netsum + ")");
		
		return act_func.activate(netsum);
	}
	
	public double activate(double[] inputs) {
		return activate(netsum(inputs));
	}
	
	// Uses the cached netsum
	public double derived_activate() {
		return act_func.derived_activate(cached_netsum);
	}
	

	// Returns the last output from the neuron
	public double output() {
		return activate(cached_netsum);
	}

	// Returns an xml representation of this neuron
	public String xml(int idx) {
		String xml = "<neuron idx="+ XML.quote(idx) +" bias=" + XML.quote(bias) + " weights=" + XML.quote(weights.length) + ">\n";
		for (int i = 0; i < weights.length; i++)
			xml += "<weight idx=" + XML.quote(i) + ">" + weights[i] + "</weight>\n";
		return xml + "</neuron>\n";
	}
}
