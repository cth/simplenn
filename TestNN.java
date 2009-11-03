import java.io.*;

public class TestNN {
	public static void main(String args[]) {
		TestNN test = new TestNN();
		//test.createAndSave();
		test.xor_net();
	}
	
    public TestNN() {
	}
	
	/* A simple application to test back-propagation */
	private void xor_net() {
		System.out.println("Testing xor net");
		
		double[][] samples = {
			{ 0.0, 0.0 },
			{ 1.0, 0.0 },
			{ 0.0, 1.0 },
			{ 1.0, 1.0 }
		};
		
		double[][] expected_outputs = {
			{ 0.0 },
			{ 1.0 },
			{ 1.0 },
			{ 0.0 }
		};
		
		FeedForwardNN nn = new FeedForwardNN(2, 1, 3, 2, 2);
		
		nn.save("xor-before-training.xml");
		
		System.out.println("Before training network");
		for (int i = 0; i < samples.length; i++) {
			//double output[] = nn.simulate(samples[i]);

			System.out.println(samples[i][0] + " xor " + samples[i][1] + " = " + nn.simulate(samples[i])[0]);
			//System.out.println(samples[i][0] + " xor " + samples[i][1] + " = " + output[0] + " " + output[1]);
			System.out.println("");
		}
		
		nn.train(samples, expected_outputs, 0.001, -1);
		
		nn.save("xor.xml");
		
		// Show simulation:
		System.out.println("After training network");		
		for (int i = 0; i < samples.length; i++) {
			//double output[] = nn.simulate(samples[i]);
			//System.out.println(samples[i][0] + " xor " + samples[i][1] + " = " + output[0] + " " + output[1]);			
			System.out.println(samples[i][0] + " xor " + samples[i][1] + " = " + nn.simulate(samples[i])[0]);
		}
		
		nn.save_graphviz("xornet.dot");
	}
	
	private void createAndSave() {
		FeedForwardNN nn = new FeedForwardNN(10, 10, 3, 10, 0.1);
		nn.save("nn1.xml");
		nn = FeedForwardNN.load("nn1.xml");
		nn.save("nn2.xml");
		
		// Now assert that nn1 and nn2 are equal
		try {
			FileInputStream f1 = new FileInputStream("nn1.xml");
			FileInputStream f2 = new FileInputStream("nn2.xml");
			DataInputStream d1 = new DataInputStream(f1);
			DataInputStream d2 = new DataInputStream(f2);
		
			String nn1 = new String();
			String nn2 = new String();
		
			while (d1.available() == 0) 
				nn1 += d1.readLine();
			
			while (d2.available() == 0)
				nn2 += d2.readLine();
				
			if (nn1.equals(nn2)) 
				System.out.println("Loading and saving NN works");
			else
				System.out.println("Loading and saving NN does NOT work!");
				
				
		} catch (IOException e) {
			System.out.println(e.toString());
		}
	}
}
