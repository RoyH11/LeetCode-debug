package sandbox.java;

public class ReadRequestHandler {


  @RestController
  class CashCardController {
    @GetMapping("/cashcards/{requestedId}")
    private ResponseEntity<CashCard> findById(@PathVariable Long requestedId) {
      CashCard cashcard = /* */
      return ResponseEntity.ok(cashcard);
    }
  }  
    
}
