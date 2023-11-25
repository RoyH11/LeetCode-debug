/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public static boolean helper(TreeNode A, TreeNode B){
        if(!(A==null && B==null)){
            if( A==null || B==null || (A.val != B.val) ){
                System.out.println("False");
                return false;
            }else{
                System.out.println("A.val: " + A.val + " B.val: " + B.val);
                return helper(A.left, B.right) && helper(A.right, B.left);
            }
        }else{
            System.out.println("True");
            return true;
        }
    }
    public static boolean isSymmetric(TreeNode root) {
        if (root == null || (root.left==null && root.right==null)){
            return true;
        }else{ 
            return helper(root.left, root.right);
        }
    }

    public static void main(String[] args) {
        // TreeNode root = new TreeNode(1);
        // TreeNode left = new TreeNode(2);
        // root.left = left;
        // TreeNode right = new TreeNode(2);
        // root.right = right;
        // TreeNode leftleft = new TreeNode(3);
        // left.left = leftleft;
        // TreeNode leftright = new TreeNode(4);
        // left.right = leftright;
        // TreeNode rightleft = new TreeNode(4);
        // right.left = rightleft;
        // TreeNode rightright = new TreeNode(3);
        // right.right = rightright;
        // System.out.println(isSymmetric(root));

        TreeNode root = new TreeNode(1);
        TreeNode left = new TreeNode(2);
        root.left = left;
        TreeNode right = new TreeNode(2);
        root.right = right;
        TreeNode leftright = new TreeNode(3);
        left.right = leftright;

        TreeNode rightright = new TreeNode(3);
        right.right = rightright;
        System.out.println(isSymmetric(root));


    }
}