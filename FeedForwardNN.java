import java.util.*;
import java.io.*;

/* A fully-connected feed-forward network with one hidden layer */
public final class FeedForwardNN {
	protected Neuron[][] layers;
	protected double learning_rate;
	protected int inputs, outputs, hidden_neurons;
	protected ActivationFunction act_func;
	
    public FeedForwardNN(int inputs, int outputs, int num_layers, int hidden_neurons, double learning_rate) {
		act_func = (ActivationFunction) new Sigmoid();

		this.learning_rate = learning_rate;
		this.inputs = inputs;
		this.outputs = outputs;
		this.hidden_neurons = hidden_neurons;

		layers = new Neuron[num_layers][];		

		for (int l = 0; l < num_layers; l++) {
			if (l == 0) 
				layers[l] = createLayer(hidden_neurons, inputs);
			else if (l == num_layers-1)
				layers[l] = createLayer(outputs, hidden_neurons);
			else
				layers[l] = createLayer(hidden_neurons, hidden_neurons);
		}
	}
	
	private FeedForwardNN () {}

	protected Neuron[] createLayer(int neurons, int inputs) {
		Neuron[] layer = new Neuron[neurons];
		Random gen = new Random();
		for (int i = 0; i < neurons; i++)
			layer[i] = new Neuron(inputs, 0, act_func);
			//layer[i] = new Neuron(inputs, gen.nextDouble(), act_func);
			
		return layer;
	}

 	/// Returns an array of outputs given an array of inputs
	public double[] simulate(double[] inputs) {
		double[] next_inputs; // used to contain the inputs to the next layer
		
		for (int l = 0; l < layers.length; l++) { // For each layer from input -> output 
			next_inputs = new double[layers[l].length];
			
			// Populate next_inputs by firing each neuron in current layer
			for (int i = 0; i < layers[l].length; i++)
				next_inputs[i] = layers[l][i].activate(inputs);
				
			inputs = next_inputs;
		}
		
		// inputs now contains the outputs from the output layer
		
		return inputs;
	}
	
	/// Run all samples through a simulation and return the average error rate
	public double evaluate(double[][] samples, double[][] desired) {
			// run simulation with all samples
			double outputs[][] = new double[samples.length][];
			
			//System.out.println("OUTPUTS:");
			for (int i = 0; i < samples.length; i++) {
				outputs[i] = simulate(samples[i]);
				/*
				System.out.println("Outputs length is " + outputs.length);
				
				for (int j = 0; j < outputs[i].length; j++)
					System.out.println("outputs[" + i + "]["+j+"]: " + outputs[i][j]);
				*/
			}
			
			// Calculate average error rate
			double total_error = 0;
			for (int i = 0; i < outputs.length; i++) {
				for (int j = 0; j < outputs[i].length; j++) {
					// Should use squared error: 0.5 * error^2 - but then, who cares, this is almost the same:
					total_error += Math.abs(outputs[i][j] - desired[i][j]);
				}
			}
					
			double average_error = total_error / (outputs.length * outputs[0].length);
			
			return average_error;
	}
	
	public void train(double[][] samples, double[][] desired, double error_margin) {
		train(samples, desired, error_margin, -1);
	}
	
	public void train(double[][] samples, double[][] desired, double error_margin, int max_epochs) {
		System.out.println("Training neural network");
		System.out.println("Allowed error: " + error_margin);
		System.out.println("Max Epochs: "+ max_epochs);
		
		
		double average_error = error_margin + 1; // average error starts out to high
		int epochs = 0;
		
		while (error_margin < average_error) {
			// Train the network on all the samples
			for (int i = 0; i < samples.length; i++)
				backprop_train(samples[i], desired[i]);
				
			// Run simulation to evaluate error
			average_error = evaluate(samples, desired);
			
			epochs++;
			if ((epochs % 100) == 0)
				System.out.println("Epoch: " + epochs + " - Average error:" + average_error);
			
			if (epochs == max_epochs)
				break;
		}
		
		System.out.println("Total epochs: " + epochs);
		
	}

	// Implementation of the back-propagation algorithm
	public void backprop_train(double[] sample, double[] desired) {

/*
		System.out.println("");
		System.out.println("Printing sample of length " + sample.length + ":");
		for (int i = 0; i < sample.length; i++)
			System.out.print(sample[i]+",");

		System.out.println("");
		
		System.out.println("Printing desired");
		for (int i = 0; i < desired.length; i++)
			System.out.print(desired[i]+ ",");
		System.out.println("");
*/
		
		double[] outputs;
		//Neuron[] output_layer = layers[layers.length-1];
		Neuron[] current_layer = layers[layers.length-1], previous_layer = layers[layers.length-2], next_layer = null;
		double[] delta = new double[current_layer.length];
		double new_delta[];
		
		// First a simulation:
		outputs = simulate(sample);
		
		// Compute gradient in the units in the output layer
		for (int j = 0; j < current_layer.length; j++)
			delta[j] = current_layer[j].derived_activate() * (desired[j] - outputs[j]);
			
		// Update error for output layers weights here
		for (int j = 0; j < current_layer.length; j++)
			for (int w = 0; w < current_layer[j].weights.length; w++)
				current_layer[j].weights[w] += learning_rate * previous_layer[w].output() * delta[j];
		
		// Process the layers backwards and propagate the error gradient
		// From layer before output-layer downto input-layer
		for (int i = layers.length - 2; i >= 0; i--) {
			next_layer = current_layer; // In first iteration, the output layer
			current_layer = layers[i]; // The layer preceeding former current_layer
			if (i != 0)
				previous_layer = layers[i-1]; // The layer preceeding current_layer
			else
				previous_layer = null; // previous layer is the input layer
				
			new_delta = new double[current_layer.length];
			
			for (int j = 0; j < current_layer.length; j++) { // For each neuron in current layer
				double total = 0;

				// Add up the weighted gradient from next layer:
				for (int k = 0; k < next_layer.length; k++)
					total += delta[k] * next_layer[k].weights[j]; 
					
				new_delta[j] = current_layer[j].derived_activate() * total;

				// Update the weights in this neuron:
				if (previous_layer == null) { // Input layer
					for (int w = 0; w < current_layer[j].weights.length; w++)
						current_layer[j].weights[w] += learning_rate * new_delta[j] * sample[w];
						
				} else { // Intermediary layer
					for (int w = 0; w < current_layer[j].weights.length; w++)
						current_layer[j].weights[w] += learning_rate * new_delta[j] * previous_layer[w].output();
				}
			}
			
			delta = new_delta;
		}
	}
	
	public static FeedForwardNN load(String filename) {
		return XML.parse(filename);
	}
	
	
	/// Save the neural network as an XML file which can be loaded using the load method
	public void save(String filename) {
		try {
			PrintStream p = new PrintStream(new FileOutputStream(filename));
			p.println(xml());
			p.close();
		} catch (IOException e) {
			System.out.println("Error writing to file: " + filename);
		}
	}
	
	public String xml() {
		String xml = new String();
		xml = xml + "<neural-net layers=" + XML.quote(layers.length) + 
				" inputs=" + XML.quote(inputs) + 
				" outputs=" + XML.quote(outputs) +
				" hidden-neurons=" + XML.quote(hidden_neurons) + 
				" learning-rate=" + XML.quote(learning_rate) + ">\n";
				
		for (int i = 0; i < layers.length; i++) {
			Neuron[] layer = layers[i];
			
			xml += "<layer idx=" + XML.quote(i) + ">\n";
			for (int n = 0; n < layer.length; n++)
				xml += layer[n].xml(n);
			xml += "</layer>\n";
		}
		xml = xml + "</neural-net>\n";
		
		return xml;
	}
	
	// Generates the network as a graphviz diagram:
	public void save_graphviz(String filename) {
		try {	
			PrintStream p = new PrintStream(new FileOutputStream(filename));
			p.println("digraph NeuralNetwork {");
			p.println("node [ shape = circle ];");
			
			// Draw all the neuron connections
			for (int l = 0; l < layers.length - 1; l++)
				for (int i = 0; i < layers[l].length; i++)
					for (int j = 0; j < layers[l+1].length; j++)
						p.println(	"L"+ l + "N" + i + 
									" -> L" + l+1 + "N" + j);
			
			p.println("}");
		} catch (IOException e) {
			System.out.println("Error writing to file: " + filename);
			
		}
	}
}
