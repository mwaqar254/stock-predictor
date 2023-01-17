class Factorial extends Thread {
	public void run() {
		int num = 6;
		int ans = num;
		System.out.println("Roll No = 061 ");
		for (int i = num - 1; i >= 1; i--) {
			ans = num * i;
			System.out.println(num + " X " + i + " = " + ans );
			num = ans;
		}
	}
}

class Square extends Thread {
	public void run() {
		int re = 1;
		int po = 6;
		for (int i = 1; i <= po; i++) {
			re *= po;
			System.out.println(po + "^" + i + " = " + re);

		}
	}

}

public class Main1 {

	public static void main(String[] args) {
		Factorial thread1 = new Factorial();
		Square thread2 = new Square();
		thread1.start();
		thread2.start();

	}

}
