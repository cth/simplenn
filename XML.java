import java.util.*;
import java.io.*;

public class XML implements DocHandler {
	public static String quote(String input) {
		return new String("\"" + input + "\"");
	}
	
	public static String quote(int inp) { return quote(""+inp); }
	public static String quote(double inp) { return quote(""+inp); }
	
	public static FeedForwardNN parse(String filename) {
		XML xml = new XML();
		try {
			QDParser.parse(xml,new FileReader(filename));
		} catch (Exception e) {
			System.out.println("Error in XML.parse: " + e.toString());
			e.printStackTrace();
		}
		return xml.nn;
	}
	
	public FeedForwardNN nn = null;
	// private variables used by the parser to aid nn construction
	private boolean expect_weight = false;
	private int current_layer = -1;
	private int current_neuron_idx = -1;
	private int current_weight_idx = -1;
	private ActivationFunction act_func = null;
	
  	public void startElement(String tag,Hashtable h) throws Exception {
		if (tag.equals("neural-net")) { 
			int layers = Integer.parseInt((String)h.get("layers"));
			int inputs = Integer.parseInt((String)h.get("inputs"));
			int outputs = Integer.parseInt((String)h.get("outputs"));
			int hidden_neurons = Integer.parseInt((String)h.get("hidden-neurons"));
			double learning_rate = Double.parseDouble((String)h.get("learning-rate"));
			nn = new FeedForwardNN(inputs, outputs, layers, hidden_neurons, learning_rate);
		} else if (tag.equals("layer")) {
			current_layer = Integer.parseInt((String)h.get("idx"));
		} else if (tag.equals("neuron")) {
			current_neuron_idx = Integer.parseInt((String)h.get("idx"));
			int weights = Integer.parseInt((String)h.get("weights"));
			double bias = Double.parseDouble((String)h.get("bias"));
			nn.layers[current_layer][current_neuron_idx] = new Neuron(	weights, bias, nn.act_func);
		} else if (tag.equals("weight")) {
			expect_weight = true;
			current_weight_idx = Integer.parseInt((String)h.get("idx"));
		}
	}
	
  	public void endElement(String tag) throws Exception {
		if (tag.equals("weight"))
			expect_weight = false;
	}
	
	public void startDocument() throws Exception {}
  	public void endDocument() throws Exception {}
	
  	public void text(String str) throws Exception {
		if (expect_weight) {
			nn.layers[current_layer][current_neuron_idx].weights[current_weight_idx] = Double.parseDouble(str);
		}
	}
}
