import java.util.*;

class Solution {
    public static boolean isValid(String s) {

        //check odd
        if(s.length()%2 != 0){
            return false;
        }

        Stack<Character> stack = new Stack<>();
        stack.push(s.charAt(0));

        for(int i = 1; i<s.length(); i++){
            char curr = s.charAt(i);
            if(!stack.empty()){
                char prev = stack.peek();

                String pair = ""+prev+curr;

                if(pair.equals("()") || pair.equals("[]") || pair.equals("{}")){
                    stack.pop();
                }else{
                    stack.push(curr);
                }
            }else{
                stack.push(curr);
            }
            
        }

        return stack.empty();
    }

    public static void main(String[] args){

        String s = "()[]{}";
        System.out.println(isValid(s));
    }
}