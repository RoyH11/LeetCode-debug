import java.util.*;

public class ListNode {
    int val;
    ListNode next;
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}
  
 
class Solution {
    
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode result_tracker;

        if(list1 ==null){
            return list2;
        }else if(list2== null){
            return list1;
        }

        if (list1.val < list2.val){
            result_tracker = new ListNode(list1.val);
            list1 = list1.next;
        }else{
            result_tracker = new ListNode(list2.val);
            list2 = list2.next;
        }

        ListNode result = result_tracker;

        while(list1 != null && list2!=null){
            if(list1==null){
                result_tracker.next = new ListNode(list2.val);
                list2 = list2.next;
            } else if(list2==null){
                result_tracker.next = new ListNode(list1.val);
                list1 = list1.next;
            }else{
                if (list1.val<list2.val){
                    result_tracker.next = new ListNode(list1.val);
                    list1 = list1.next;
                }else{
                    result_tracker.next = new ListNode(list2.val);
                    list2 = list2.next;
                }
            }
            result_tracker = result_tracker.next;
        }

        return result.next;
    }

    public static void main(String[] args){

        String s = "()[]{}";
        System.out.println(isValid(s));
    }
}