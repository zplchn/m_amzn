from typing import Optional
from enum import Enum
from abc import ABC


class Display:
    # This class is given with implementation. And the ONLY one
    @staticmethod
    def read_input(display_text: str) -> str:
        pass

    @staticmethod
    def show_error(err_msg: str) -> None:
        pass

    @staticmethod
    def show_message(msg: str) -> None:
        pass

'''
Implement methods below here
'''

class AccountStoreRespCode(Enum):
    USER_NOT_EXIST = 0
    USER_ALREADY_EXIST = 1

class AccountStore:
    @staticmethod
    def check_user(name: str, address: str) -> AccountStoreRespCode:
        return AccountStoreRespCode.USER_NOT_EXIST

    @staticmethod
    def create_user(name: str, address: str, pin: str) -> Optional[int]:
        pass

    @staticmethod
    def verify_user(uid: int, pin: str) -> bool:
        pass

    @staticmethod
    def withdraw(uid: int, amount: int) -> bool:
        pass

    @staticmethod
    def get_balance(uid: int) -> int:
        balance = 1000
        # balance = Account.get_balance_for_user(uid)
        return balance

    @staticmethod
    def set_balance(uid: int, amount: int) -> None:
        # balance = Account.set_balance_for_user(uid)
        return


class CardHandler:
    @staticmethod
    def create_card(uid: int):
        pass

    @staticmethod
    def read_card() -> Optional[int]:
        pass

    @staticmethod
    def return_card() -> None:
        pass

    @staticmethod
    def lock_card(num):
        pass


class CashDispenser:
    @staticmethod
    def dispense(amount: int) -> bool:
        pass


class CashInTake:
    @staticmethod
    def take() -> int:
        pass

'''
State design pattern: Actions changes according to internal state, as if the implementation changes dynamically
'''


class ATMState(Enum):
    NO_CARD = 0
    HAS_CARD = 1
    AUTHORIZED = 2
    MAIN_MENU = 3

    WITHDRAW_SELECTED = 10
    WITHDRAW_AMOUNT_ENTERED = 11
    WITHDRAW_LOW_BALANCE = 13

    DEPOSIT_SELECTED = 20
    DEPOSIT_ACCEPTED = 21


class ATM:
    def __init__(self):
        self.state_obj = None
        self.withdraw_amount = None
        self.uid = None
        self.card = 0
        self.pin_error = 0

    states_map = {
        ATMState.NO_CARD: lambda: NoCardState(),
        ATMState.HAS_CARD: lambda: HasCardState(),
    }

    def set_state(self, state: ATMState) -> None:
        self.state_obj = ATM.states_map[state]().set_atm(self)

    def set_withdraw_amount(self, amount) -> None:
        self.withdraw_amount = amount

    def get_withdraw_amount(self) -> int:
        return self.withdraw_amount

    def get_balance(self) -> int:
        return AccountStore.get_balance(self.uid)

    def set_balance(self, amount: int) -> None:
        AccountStore.set_balance(self.uid, amount)

    def create_account(self):
        name = Display.read_input('Enter your name:')
        address = Display.read_input('Enter your address:')

        if AccountStore.check_user(name, address) == AccountStoreRespCode.USER_ALREADY_EXIST:
            Display.show_error('User already exist!')
            return

        pin = Display.read_input('Enter pin:')

        uid = AccountStore.create_user(name, address, pin)
        if not uid:
            return

        CardHandler.create_card(uid)

    def log_in_user(self, pin: str):
        self.uid = AccountStore.verify_user(self.card, pin)
        return self.uid

    def insert_card(self):
        self.state_obj.insert_card()

    def ask_pin(self) -> int:
        return self.state_obj.ask_pin()

    def get_input(self):
        self.state_obj.get_input()

    def ask_deposit_money(self):
        self.state_obj.ask_deposit_money()

    def ask_withdraw_amount(self):
        self.state_obj.ask_withdraw_amount()

    def lock_card(self):
        CardHandler.lock_card(self.card)

    def withdraw(self):
        self.state_obj.withdraw()

    def show_menu(self):
        self.state_obj.show_menu()


class BaseState(ABC):
    '''
    Class
    1. take action
    2. store intermediate result to atm
    3. change atm to next possible state
    '''

    def __init__(self, atm: ATM = None):
        self.atm = atm

    def set_atm(self, atm: ATM):
        self.atm = atm
        return self

    def insert_card(self):
        pass

    def ask_pin(self) -> int:
        pass

    def ask_withdraw_amount(self):
        pass

    def get_input(self) -> int:
        pass

    def ask_deposit_money(self):
        pass

    def withdraw(self):
        pass

    def show_menu(self):
        pass


class NoCardState(BaseState):
    def insert_card(self) -> None:
        card = CardHandler.read_card()
        if not card:
            Display.show_error('Card cannot be read')
        self.atm.card = card
        self.atm.set_state(ATMState.HAS_CARD)


class HasCardState(BaseState):
    def ask_pin(self) -> int:
        pin = Display.read_input('Enter pin code:')
        uid = self.atm.log_in_user(pin)
        if uid:
            self.atm.set_state(ATMState.MAIN_MENU)
            return 0
        else:
            self.atm.pin_error += 1
            if self.atm.pin_error == 3:
                self.atm.lock_card()
                self.atm.set_state(ATMState.NO_CARD)
            else:
                self.atm.set_state(ATMState.HAS_CARD)
            return -1


class MainMenuState(BaseState):
    def show_menu(self) -> int:
        mode = int(Display.read_input('Choose 1. Withdraw 2. Deposit 3. Log out'))
        if mode == 1:
            self.atm.set_state(ATMState.WITHDRAW_SELECTED)
        elif mode == 2:
            self.atm.set_state(ATMState.DEPOSIT_SELECTED)
        elif mode == 3:
            CardHandler.return_card()
            self.atm.set_state(ATMState.NO_CARD)
        return mode


class WithdrawSelected(BaseState):
    def ask_withdraw_amount(self):
        amount = int(Display.read_input('Enter withdraw amount'))
        self.atm.set_withdraw_amount(amount)
        self.atm.set_state(ATMState.WITHDRAW_AMOUNT_ENTERED)


class WithdrawAmountEntered(BaseState):
    def withdraw(self) -> int:
        balance = self.atm.get_balance()
        if self.atm.withdraw_amount <= balance:
            CashDispenser.dispense(self.atm.withdraw_amount)
            self.atm.set_balance(balance - self.atm.withdraw_amount)
            self.atm.set_state(ATMState.MAIN_MENU)
            return 0
        else:
            Display.show_error('Balance is not enough')
            self.atm.set_state(ATMState.WITHDRAW_LOW_BALANCE)
            return -1


class WithdrawLowBalance(BaseState):
    def get_input(self):
        self.atm.set_state(ATMState.MAIN_MENU)


class DepositSelected(BaseState):
    def ask_deposit_money(self):
        Display.show_message('Deposit money')
        amount = CashInTake.take()
        self.atm.set_balance(self.atm.get_balance() + amount)
        self.atm.set_state(ATMState.MAIN_MENU)


def main():
    atm = ATM()
    atm.insert_card()
    i = 0
    res = -1
    while i < 3:
        if atm.ask_pin() < 0:
            i += 1
        else:
            break
    if i == 3:
        return
    mode = atm.get_input()
    if mode == 1:
        atm.ask_withdraw_amount()
        res = atm.withdraw()
        if res == 0:
            atm.show_menu()

# It's the user input to drive ATM jump between different states
if __name__ == '__main__':
    main()

























