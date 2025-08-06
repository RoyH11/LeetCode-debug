public class Parameter {
    public static void main(String[] args) {

        int x = 5;
        System.out.println("x in main before pf: " + x);
        pf(x);
        System.out.println("x in main: " + x);

        System.out.println();

        MyObject myObject = new MyObject(10);
        System.out.println("myObject in main before of: " + myObject.getValue());
        of(myObject);
        System.out.println("myObject in main: " + myObject.getValue());
    }

    public static void pf(int x) {
        x = x + 1;
        System.out.println("x in pf: " + x);
    }

    public static void of(MyObject myObject) {
        myObject.setValue(myObject.getValue() + 1);
        System.out.println("MyObject in of: " + myObject.getValue());
    }

}
