
public class Wilder {
    public static void main(String[] args) {
        System.out.println("What is up Wilder!");

        System.out.println("Choose your fate: ");
        System.out.println("1. trap in the program forever");
        System.out.println("2. exit the program");

        // Simulate user input for demonstration purposes
        int choice = -1;

        while (choice < 1 || choice > 2) {
            try {
                choice = Integer.parseInt(System.console().readLine("Enter your choice (1 or 2): "));
            } catch (NumberFormatException e) {
                System.out.println("Invalid input. Please enter 1 or 2.");
            }
        }

        if (choice == 1) {
            while (true) {
                System.out.println("You are trapped in the program forever!");
                // Simulate some processing
                try {
                    Thread.sleep(1000); // Sleep for 1 second
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        } else {
            System.out.println("Exiting the program. Goodbye!");
        }


        
    }
}
