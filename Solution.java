import java.util.*;

class Solution {
    public boolean wordPattern(String pattern, String s) {

        


        String[] words = s.split(" ");
        HashMap<Character, String> map = new HashMap<>();
        ArrayList<Character> patternLetters = new ArrayList<>();

        //check if words and pattern is same length
        if(words.length!= pattern.length()){
            return false;
        }

        //get unique letters
        for(int i=0; i<pattern.length(); i++){
            if(! patternLetters.contains(pattern.charAt(i))){
                patternLetters.add(pattern.charAt(i));
            }
        }


        // construct map
        int current = 0;
        for (String word: words){
            if(! map.containsValue(word)){
                if(current >= patternLetters.size()){
                    return false;
                }
                map.put(patternLetters.get(current), word);
                current++;
            }
        }

        

        for(int i=0; i<pattern.length(); i++){
            if(map.get(pattern.charAt(i)) == null){
                return false;
            }
            if(! map.get(pattern.charAt(i)).equals(words[i])){
                return false;
            }
        }
        return true; 
    }

    public static void main(String[] args){
        Solution s = new Solution();
        System.out.println(s.wordPattern("abba", "dog cat cat dog"));
    }
}