import java.util.*;

class Solution {
    public static List<String> summaryRanges(int[] nums) {
        List<String> result = new ArrayList<>();
        if(nums.length > 0){
            for(int i=0; i<nums.length; i++){
                
                String current = Integer.toString(nums[i]);                
                

                if( ((i+1)<nums.length) && ((nums[i]+1) == nums[i+1]) ){
                    
                    // continuous
                    current = current+"->";
                    for(int j = i+1; j<nums.length; j++){
                        
                        if(nums[j]!=nums[j-1]+1){
                            // continuous 
                            current = current+ Integer.toString(nums[j-1]);
                            i = j-1;
                            result.add(current);
                            break;
                        }
                    }
                    if(current.charAt(current.length()-1) == '>'){
                        current = current+ Integer.toString(nums[nums.length-1]);
                        result.add(current);
                        break;
                    }
                }else{
                    
                    // end single case or discontinuous
                    result.add(current);
                }
            }
        }
        return result;
    }

    public static void main(String[] args){

        int[] nums2 = {0,1,2,4,5,7};
        System.out.println(summaryRanges(nums2));
        int[] nums3 = {0,2,3,4,6,8,9};
        System.out.println(summaryRanges(nums3));
    }
}