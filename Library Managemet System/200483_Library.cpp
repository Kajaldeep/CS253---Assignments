#include <bits/stdc++.h>
#include <fstream>
#include <conio.h>
using namespace std;

class Book;
class User;
class Professor;
class Librarian;
class Student;

class User
{
    friend void uservec();
    friend void login(string);

public:
    string username;
    string ID;
    string role;
    string password;
    void show(Book *);
    void booklist();
    string search();
};

class Professor : public User
{
    vector<Book> books;
    template <typename T>
    friend void issuedbooks(T *);
    template <typename T>
    friend void rtrn(T *);

public:
    int Fine_amount;
    int period;
    int amount;
    Professor()
    {
        Fine_amount = 0;
        period = 60;
        amount = 5;
    }
    void prof();
};

class Student : public User
{
    vector<Book> books;
    template <typename T>
    friend void issuedbooks(T *);
    template <typename T>
    friend void rtrn(T *);

public:
    int Fine_amount;
    int period;
    int amount;
    Student()
    {
        Fine_amount = 0;
        period = 30;
        amount = 2;
    }
    void stud();
};

class Librarian : public User
{
public:
    void lib();
    void userlist();
    void modify();
    void modifyuser();
    void addbook();
};

class Book
{
    friend void bookvec();
    template <typename T>
    friend void rtrn(T *);

public:
    string Title;
    string Author;
    string ISBN;
    string Publication;
    string Available;
    string Borrower;
    string ID;
    string Duedate;
    Book()
    {
        Available = "1";
        Borrower = "0";
        ID = "0";
        Duedate = "0";
    }
    void Book_Request();
    void Show_duedate();
};

vector<Librarian> libv;
vector<Student> studv;
vector<Professor> profv;
vector<Book> bookv;
vector<vector<string>> vecb;
vector<vector<string>> vecu;

void Select_role();
void logout();

vector<vector<string>> openbook()
{
    string line, word;
    vector<vector<string>> col;
    ifstream myfile("Books.csv");
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            stringstream ss(line);
            vector<string> row;
            while (getline(ss, word, ','))
            {
                row.push_back(word);
            }
            col.push_back(row);
        }
        myfile.close();
    }
    return col;
}

vector<vector<string>> openuser()
{
    string line, word;
    vector<vector<string>> col;
    ifstream myfile("User.csv");
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            stringstream ss(line);
            vector<string> row;
            while (getline(ss, word, ','))
            {
                row.push_back(word);
            }
            col.push_back(row);
        }
        myfile.close();
    }
    return col;
}

void bookvec()
{
    Book obj;
    int i;
    bookv.clear();
    for (i = 0; i < vecb.size(); i++)
    {
        obj.Title = vecb[i][0];
        obj.Author = vecb[i][1];
        obj.ISBN = vecb[i][2];
        obj.Publication = vecb[i][3];
        obj.Available = vecb[i][4];
        obj.Borrower = vecb[i][5];
        obj.ID = vecb[i][6];
        obj.Duedate = vecb[i][7];
        bookv.push_back(obj);
    }
    return;
}

void uservec()
{
    Professor obj1;
    Student obj2;
    Librarian obj3;
    profv.clear();
    studv.clear();
    libv.clear();
    int i;
    for (i = 0; i < vecu.size(); i++)
    {
        if (vecu[i][3] == "1")
        {
            obj1.username = vecu[i][0];
            obj1.ID = vecu[i][1];
            obj1.password = vecu[i][2];
            profv.push_back(obj1);
        }
        else if (vecu[i][3] == "2")
        {
            obj2.username = vecu[i][0];
            obj2.ID = vecu[i][1];
            obj2.password = vecu[i][2];
            studv.push_back(obj2);
        }
        else
        {
            obj3.username = vecu[i][0];
            obj3.ID = vecu[i][1];
            obj3.password = vecu[i][2];
            libv.push_back(obj3);
        }
    }
}

template <typename T>
void copy(User *user1, T *user2)
{
    user2->username = user1->username;
    user2->password = user1->password;
    user2->role = user1->role;
    user2->ID = user1->ID;
    return;
}

void Librarian::addbook()
{
    Book book1;
    string str;
    cout << "\n\t\tEnter Book's Title : ";
    cin >> book1.Title;
    str = str + book1.Title + ",";
    cout << "\n\t\tEnter Book's Author : ";
    cin >> book1.Author;
    str = str + book1.Author + ",";
    cout << "\n\t\tEnter Book's ISBN : ";
    cin >> book1.ISBN;
    str = str + book1.ISBN + ",";
    cout << "\n\t\tEnter Book's Publication : ";
    cin >> book1.Publication;
    str = str + book1.Publication + ",";
    book1.Available = "1";
    str = str + "1";
    str = str + ",0,0,0,0";
    bookv.push_back(book1);
    ofstream myfile;
    myfile.open("Books.csv", ios::app);
    myfile << str;
    myfile << endl;
    myfile.close();
    return;
}

template <typename T>
void issuedbooks(T *obj)
{
    ((*obj).books).clear();
    int i;
    for (i = 0; i < bookv.size(); i++)
    {
        if (bookv[i].ID == (*obj).ID)
        {
            ((*obj).books).push_back(bookv[i]);
        }
    }
}

void User::show(Book *obj)
{
    cout << "\n\t\tTitle : " << (*obj).Title << endl;
    cout << "\n\t\tAuthor : " << (*obj).Author << endl;
    cout << "\n\t\tISBN : " << (*obj).ISBN << endl;
    cout << "\n\t\tPublication : " << (*obj).Publication << endl;
    if ((*obj).Available == "1")
    {
        cout << "\n\t\tAvailable : "
             << "Yes" << endl;
    }
    else
    {
        cout << "\n\t\tAvailable : "
             << "No" << endl;
        cout << "\n\t\tDuedate : " << (*obj).Duedate << endl;
    }
    return;
}

string User::search()
{
    int j, k;
    string str1;
    cout << "\n\t\tPlease Choose One Option :\n";
    cout << "\n\t\t1.By Book's Name\n\n\t\t2.By Book's ISBN\n";
    cout << "\n\t\tEnter Your Choice : ";
    cin >> j;
    if (j == 1)
    {
        cout << "\n\t\tEnter Book's Name : ";
        cin >> str1;
        for (k = 0; k < bookv.size(); k++)
        {
            if (bookv[k].Title == str1)
            {
                return to_string(k) + "," + to_string(j) + "," + str1;
            }
        }
    }
    else
    {
        cout << "\n\t\tEnter Book's ISBN : ";
        cin >> str1;
        for (k = 0; k < bookv.size(); k++)
        {
            if (bookv[k].ISBN == str1)
            {
                return to_string(k) + "," + to_string(j) + "," + str1;
            }
        }
    }
}

string duedate(string str, int period)
{
    int d, m, y, dd, mm, yy, dur = 0, j;
    string date, month;
    const int monthdays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    d = stoi(str.substr(0, 2));
    m = stoi(str.substr(3, 2));
    y = stoi(str.substr(6, 2));
    j = m;
    dur = monthdays[m - 1] - d + 1;
    j++;
    while ((dur + monthdays[j - 1]) < period)
    {
        dur = dur + monthdays[j - 1];
        if (j != 12)
            j++;
        else
        {
            j = 1;
        }
    }
    dd = period - dur + 1;
    mm = j;
    if (m < mm)
    {
        yy = y;
    }
    else
    {
        yy = y + 1;
    }
    date = to_string(dd);
    if (date.length() == 1)
    {
        date = "0" + date;
    }
    month = to_string(mm);
    if (month.length() == 1)
    {
        month = "0" + month;
    }
    return date + "/" + month + "/" + to_string(yy);
}

int calc_fine(string date, string str1, int amount)
{
    int fine = 0, dur = 0;
    string str = date;
    int d, m, y, dd, mm, yy;
    d = stoi(str.substr(0, 2));
    m = stoi(str.substr(3, 2));
    y = stoi(str.substr(6, 2));
    dd = stoi(str1.substr(0, 2));
    mm = stoi(str1.substr(3, 2));
    yy = stoi(str1.substr(6, 2));

    const int monthDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (y == yy)
    {
        if (m == mm)
        {
            if (dd == d)
            {
                return 0;
            }
            else
            {
                return ((dd - d) * amount);
            }
        }
        else if (mm > m)
        {
            dur = dd + monthDays[m - 1] - d + 1;
            for (int j = (m + 1); j < mm; j++)
            {
                dur = dur + monthDays[j - 1];
            }
            return (amount * (dur));
        }
    }
    else
    {
        dur = dd + monthDays[m - 1] - d + 1;
        for (int j = (m + 1); j < mm; j++)
        {
            dur = dur + monthDays[j - 1];
            if (j == 12)
            {
                j = 0;
            }
        }
        return (amount * (dur));
    }
}

template <typename T>
void clear_fine(T *obj)
{
    int i;
    (*obj).Fine_amount = 0;
    string date, line;
    cout << "\n\t\tEnter today's date : ";
    cin >> date;
    ofstream fileout;
    fileout.open("temp.csv", ios::app);
    for (i = 0; i < vecb.size(); i++)
    {
        if (vecb[i][5] == (*obj).username)
        {
            fileout << vecb[i][0] << "," << vecb[i][1] << "," << vecb[i][2] << "," << vecb[i][3] << ","
                    << vecb[i][4] << "," << vecb[i][5] << "," << vecb[i][6] << "," << date;
            fileout << endl;
        }
        else
        {
            fileout << vecb[i][0] << "," << vecb[i][1] << "," << vecb[i][2] << "," << vecb[i][3] << ","
                    << vecb[i][4] << "," << vecb[i][5] << "," << vecb[i][6] << "," << vecb[i][7];
            fileout << endl;
        }
    }
    fileout.close();
    remove("Books.csv");
    rename("temp.csv", "Books.csv");
    vecb = openbook();
    bookvec();
}

void User::booklist()
{
    system("cls");
    int i;
    if (bookv.size() != 0)
    {
        cout << "\n\t\tHere is the list of books :- \n";
        for (i = 0; i < bookv.size(); i++)
        {
            show(&(bookv[i]));
            cout << "-------------------------------------------------";
        }
    }
    else
    {
        cout << "\n\t\tSorry, No books available !\n"
             << endl;
    }
}

void Librarian::userlist()
{
    system("CLS");
    int i;
    cout << "\n\t---PROFESSORS---\n";
    for (i = 0; i < profv.size(); i++)
    {
        cout << "\n\tUsername : " << profv[i].username << endl;
        cout << "\n\t\tID : " << profv[i].ID << endl;
        cout << "-------------------------------------------------";
    }
    cout << "\n\t---STUDENTS---\n";
    for (i = 0; i < studv.size(); i++)
    {
        cout << "\n\tUsername : " << studv[i].username << endl;
        cout << "\n\t\tID : " << studv[i].ID << endl;
        cout << "-------------------------------------------------";
    }
    cout << "\n\t---LIBRARIANS---\n";
    for (i = 0; i < libv.size(); i++)
    {
        cout << "\n\tUsername : " << libv[i].username << endl;
        cout << "\n\t\tID : " << libv[i].ID << endl;
        cout << "-------------------------------------------------";
    }
}

template <typename T>
void issue(T *obj)
{
    int available, j;
    string line, word;
    system("cls");
    string name, id, date;
    cout << "\n\t\t->Please Enter Details :-\n";
    cout << "\n\t\tEnter Book Name : ";
    cin >> name;
    for (j = 0; j < bookv.size(); j++)
    {
        if (bookv[j].Title == name)
        {
            available = stoi(bookv[j].Available);
            break;
        }
    }
    if (available == 1)
    {
        int i = 0;
        cout << "\n\t\tEnter date (dd/mm/yy): ";
        cin >> date;
        bookv[j].Available = "0";
        bookv[j].Borrower = (*obj).username;
        bookv[j].ID = (*obj).ID;
        bookv[j].Duedate = duedate(date, (*obj).period);
        ofstream fileout;
        fileout.open("temp.csv", ios::app);
        for (i = 0; i < vecb.size(); i++)
        {
            if (vecb[i][0] == name)
            {
                fileout << vecb[i][0] << "," << vecb[i][1] << "," << vecb[i][2] << "," << vecb[i][3] << ","
                        << "0"
                        << "," << (*obj).username << "," << (*obj).ID << "," << bookv[j].Duedate;
                fileout << endl;
            }
            else
            {
                fileout << vecb[i][0] << "," << vecb[i][1] << "," << vecb[i][2] << "," << vecb[i][3] << ","
                        << vecb[i][4] << "," << vecb[i][5] << "," << vecb[i][6] << "," << vecb[i][7];
                fileout << endl;
            }
        }
        fileout.close();
        remove("Books.csv");
        rename("temp.csv", "Books.csv");
        cout << "\n\t\tIssued Successfully !!\n"
             << endl;
        vecb = openbook();
    }
    else
    {
        cout << "\n\t\tNot Available !!\n"
             << endl;
    }
}

template <typename T>
void rtrn(T *obj)
{
    int i;
    string date1, name, str1;
    cout << "\n\t\t->Please Enter Details :-\n";
    cout << "\n\t\tEnter Book Name : ";
    cin >> name;
    cout << "\n\t\tEnter today's date (dd/mm/yy) : ";
    cin >> str1;
    ofstream fileout;
    fileout.open("temp.csv", ios::app);
    for (i = 0; i < vecb.size(); i++)
    {
        if (vecb[i][0] == name)
        {
            date1 = vecb[i][7];
            fileout << vecb[i][0] << "," << vecb[i][1]
                    << "," << vecb[i][2]
                    << "," << vecb[i][3]
                    << ","
                    << "1"
                    << ","
                    << "0"
                    << ","
                    << "0"
                    << ","
                    << "0"
                    << ","
                    << "0";
            fileout << endl;
        }
        else
        {
            fileout << vecb[i][0] << "," << vecb[i][1] << "," << vecb[i][2] << "," << vecb[i][3] << ","
                    << vecb[i][4] << "," << vecb[i][5] << "," << vecb[i][6] << "," << vecb[i][7];
            fileout << endl;
        }
    }
    fileout.close();
    remove("Books.csv");
    rename("temp.csv", "Books.csv");
    cout << "\n\t\tBook returned successfully!!\n";
    for (int k = 0; k < bookv.size(); k++)
    {
        if (bookv[k].Title == name)
        {
            bookv[k].Available = "1";
            bookv[k].Borrower = "0";
            bookv[k].ID = "0";
            bookv[k].Duedate = "0";
        }
        break;
    }
    for (int k = 0; k < ((*obj).books).size(); k++)
    {
        if ((*obj).books[k].Title == name)
        {
            ((*obj).books).erase(((*obj).books).begin() + k);
            break;
        }
    }
    (*obj).Fine_amount = (*obj).Fine_amount - calc_fine(date1, str1, (*obj).amount);
    vecb = openbook();
}

void Librarian::modify()
{
    system("cls");
    int i, comp, k;
    string str2, str1;
    vector<string> v;
    cout << "\n\t\t>>Please Choose One Option:\n";
    cout << "\n\t\t1.Add new book\n\n\t\t2.Update book\n\n\t\t3.Delete a Book\n\n\t\t4.Search book\n\n\t\t5.Go back\n\n\t\t6.Close Application\n";
    cout << "\n\t\tEnter your choice : ";
    cin >> i;
    if (i == 2 || i == 3 || i == 4)
    {
        stringstream ss(search());
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ',');
            v.push_back(substr);
        }
        k = stoi(v[0]);
        str1 = v[2];
        comp = v[1] == "1" ? 0 : 2;
    }
    switch (i)
    {
    case 1:
    {
        addbook();
        cout << "\n\t\tBook added successfully!!\n"
             << endl;
        break;
    }
    case 2:
    {
        cout << "\n\t\tBook's Title : ";
        cin >> str2;
        bookv[k].Title = str2;
        cout << "\n\t\tBook's Author : ";
        cin >> str2;
        bookv[k].Author = str2;
        cout << "\n\t\tBook's ISBN : ";
        cin >> str2;
        bookv[k].ISBN = str2;
        cout << "\n\t\tBook's Publication : ";
        cin >> str2;
        bookv[k].Publication = str2;
        ofstream outfile;
        outfile.open("temp.csv", ios::app);
        for (int a = 0; a < vecb.size(); a++)
        {
            if (vecb[a][0] == bookv[k].Title)
            {
                outfile << bookv[k].Title << "," << bookv[k].Author << "," << bookv[k].ISBN << "," << bookv[k].Publication << "," << bookv[k].Available << "," << bookv[k].Borrower << "," << bookv[k].ID << "," << bookv[k].Duedate;
                outfile << endl;
            }
            else
            {
                outfile << vecb[a][0] << "," << vecb[a][1] << "," << vecb[a][2] << "," << vecb[a][3] << ","
                        << vecb[a][4] << "," << vecb[a][5] << "," << vecb[a][6] << "," << vecb[a][7];
                outfile << endl;
            }
        }
        outfile.close();
        remove("Books.csv");
        rename("temp.csv", "Books.csv");
        cout << "\n\t\tBook updated successfully !!\n"
             << endl;
        vecb = openbook();
        break;
    }
    case 3:
    {
        bookv.erase(bookv.begin() + k);
        ofstream outfile;
        outfile.open("temp.csv", ios::app);
        for (int a = 0; a < vecb.size(); a++)
        {
            if (vecb[a][comp] != str1)
            {
                outfile << vecb[a][0] << "," << vecb[a][1] << "," << vecb[a][2] << "," << vecb[a][3] << ","
                        << vecb[a][4] << "," << vecb[a][5] << "," << vecb[a][6] << "," << vecb[a][7];
                outfile << endl;
            }
        }
        outfile.close();
        remove("Books.csv");
        rename("temp.csv", "Books.csv");
        vecb = openbook();
        break;
    }
    case 4:
    {
        if (k != bookv.size())
        {
            cout << "\n\t\tBook Found :-\n";
            show(&(bookv[k]));
            break;
        }
        else
        {
            cout << "\n\t\tBook Not Found :(\n";
        }
    }
    case 5:
    {
        system("cls");
        lib();
        break;
    }
    case 6:
        exit(0);
    }
}

void Librarian::modifyuser()
{
    User user1;
    system("cls");
    string str = "", del;
    int i, j, k, comp, idx, comp1;
    cout << "\n\t\t>>Please Choose One Option:\n";
    cout << "\n\t\t1.Add an User\n\n\t\t2.Update user\n\n\t\t3.Delete an user\n\n\t\t4.Search user\n\n\t\t5.Go back\n\n\t\t6.Close Application\n";
    cout << "\n\t\tEnter your choice : ";
    cin >> i;
    system("cls");
    if (i != 5 && i != 6)
    {
        cout << "\n\t\t>>Select user type:\n";
        cout << "\n\t\t1.Professor\n\n\t\t2.Student\n\n\t\t3.Librarian\n";
        cout << "\n\t\tEnter your choice : ";
        cin >> j;
        if (i != 1)
        {
            cout << "\n\t\tPlease Choose One Option:-\n";
            cout << "\n\t\t1.By Username\n\n\t\t2.By ID\n";
            cout << "\n\t\tEnter Your Choice : ";
            cin >> k;
            if (k == 1)
            {
                cout << "\n\t\tEnter Username : ";
                cin >> del;
                comp = 0;
                comp1 = 5;
                if (j == 1)
                {
                    Professor user2;
                    for (int a = 0; a < profv.size(); a++)
                    {
                        if (del == profv[a].username)
                        {
                            idx = a;
                            break;
                        }
                    }
                }
                else if (j == 2)
                {
                    Student user2;
                    for (int a = 0; a < studv.size(); a++)
                    {
                        if (del == studv[a].username)
                        {
                            idx = a;
                            break;
                        }
                    }
                }
                else if (j == 3)
                {
                    Librarian user2;
                    for (int a = 0; a < libv.size(); a++)
                    {
                        if (del == libv[a].username)
                        {
                            idx = a;
                            break;
                        }
                    }
                }
            }
            else
            {
                cout << "\n\t\tEnter ID : ";
                cin >> del;
                comp = 1;
                comp1 = 6;
                if (j == 1)
                {
                    for (int a = 0; a < profv.size(); a++)
                    {
                        if (del == profv[a].ID)
                        {
                            idx = a;
                            break;
                        }
                    }
                }
                else if (j == 2)
                {
                    for (int a = 0; a < studv.size(); a++)
                    {
                        if (del == studv[a].ID)
                        {
                            idx = a;
                            break;
                        }
                    }
                }
                else if (j == 3)
                {
                    for (int a = 0; a < libv.size(); a++)
                    {
                        if (del == libv[a].ID)
                        {
                            idx = a;
                            break;
                        }
                    }
                }
            }
        }
    }
    switch (i)
    {
    case 1:
    {
        cout << "\n\t\tEnter Username : ";
        cin >> user1.username;
        str = str + user1.username + ",";
        cout << "\n\t\tEnter ID : ";
        cin >> user1.ID;
        str = str + user1.ID + ",";
        cout << "\n\t\tEnter Password : ";
        cin >> user1.password;
        str = str + user1.password + ",";
        user1.role = to_string(j);
        str = str + user1.role;
        ofstream myfile;
        myfile.open("User.csv", std::ios::app);
        myfile << str;
        myfile << endl;
        myfile.close();
        cout << "\n\t\tUser added successfully!!\n"
             << endl;
        if (j == 1)
        {
            Professor user2;
            copy(&user1, &user2);
            profv.push_back(user2);
        }
        else if (j == 2)
        {
            Student user2;
            copy(&user1, &user2);
            studv.push_back(user2);
        }
        else
        {
            Librarian user2;
            copy(&user1, &user2);
            libv.push_back(user2);
        }
        break;
    }
    case 2:
    {
        User obj;
        string str2;
        cout << "\n\t\tRename Username : ";
        cin >> str2;
        obj.username = str2;
        cout << "\n\t\tEnter ID : ";
        cin >> str2;
        obj.ID = str2;
        cout << "\n\t\tReset Password : ";
        cin >> str2;
        obj.password = str2;
        obj.role = to_string(j);
        if (j == 1)
        {
            Professor user2;
            copy(&obj, &user2);
            profv[idx] = user2;
        }
        else if (j == 2)
        {
            Student user2;
            copy(&obj, &user2);
            studv[idx] = user2;
        }
        else
        {
            Librarian user2;
            copy(&obj, &user2);
            libv[idx] = user2;
        }
        ofstream outfile;
        outfile.open("temp.csv", ios::app);
        for (int a = 0; a < vecu.size(); a++)
        {
            if (vecu[a][comp] != del)
            {
                outfile << vecu[a][0] << "," << vecu[a][1] << "," << vecu[a][2] << "," << vecu[a][3];
                outfile << endl;
            }
            else
            {
                outfile << obj.username << "," << obj.ID << "," << obj.password << "," << obj.role;
                outfile << endl;
            }
        }
        outfile.close();
        remove("User.csv");
        rename("temp.csv", "User.csv");

        ofstream outfile1;
        outfile1.open("temp.csv", ios::app);
        for (int a = 0; a < vecb.size(); a++)
        {
            if (vecb[a][comp1] == del)
            {
                outfile1 << vecb[a][0] << "," << vecb[a][1] << "," << vecb[a][2] << "," << vecb[a][3] << "," << vecb[a][4] << "," << obj.username << "," << obj.ID << "," << vecb[a][7];
                outfile1 << endl;
            }
            else
            {
                outfile1 << vecb[a][0] << "," << vecb[a][1] << "," << vecb[a][2] << "," << vecb[a][3] << ","
                         << vecb[a][4] << "," << vecb[a][5] << "," << vecb[a][6] << "," << vecb[a][7];
                outfile1 << endl;
            }
        }
        outfile1.close();
        remove("Books.csv");
        rename("temp.csv", "Books.csv");
        vecu = openuser();
        vecb = openbook();
        bookvec();
        uservec();
        break;
    }
    case 3:
    {
        ofstream outfile;
        outfile.open("temp.csv", ios::app);
        for (int a = 0; a < vecu.size(); a++)
        {
            if (vecu[a][comp] != del)
            {
                outfile << vecu[a][0] << "," << vecu[a][1] << "," << vecu[a][2] << "," << vecu[a][3];
                outfile << endl;
            }
        }
        outfile.close();
        remove("User.csv");
        rename("temp.csv", "User.csv");
        vecu = openuser();
        if (j == 1)
        {
            profv.erase(profv.begin() + idx);
        }
        else if (j == 2)
        {

            studv.erase(studv.begin() + idx);
        }
        else
        {

            libv.erase(libv.begin() + idx);
        }
        break;
    }
    case 4:
    {
        if (j == 1)
        {
            if (idx != profv.size())
            {
                cout << "\n\t\tUser Found :-\n";
                cout << "\n\t\tUsername : " << profv[idx].username << endl;
                cout << "\n\t\tID : " << profv[idx].ID << endl;
            }
            else
            {
                cout << "\n\t\tUser Not Found :-\n";
            }
        }
        else if (j == 2)
        {
            if (idx != studv.size())
            {
                cout << "\n\t\tUser Found :-\n";
                cout << "\n\t\tUsername : " << studv[idx].username << endl;
                cout << "\n\t\tID : " << studv[idx].ID << endl;
            }
            else
            {
                cout << "\n\t\tUser Not Found :-\n";
            }
        }
        else
        {
            if (idx != libv.size())
            {
                cout << "\n\t\tUser Found :-\n";
                cout << "\n\t\tUsername : " << libv[idx].username << endl;
                cout << "\n\t\tID : " << libv[idx].ID << endl;
            }
            else
            {
                cout << "\n\t\tUser Not Found :-\n";
            }
        }
        break;
    }
    case 5:
        lib();
        break;
    case 6:
        exit(0);
        break;
    default:
        exit(0);
        break;
    }
}

void Professor::prof()
{
    int i;
    cout << "\n\t************ WELCOME ************\n";
    cout << "\n\t\t>>Please Choose One Option:\n";
    cout << "\n\t\t1.View BookList\n\n\t\t2.View Your Issued Books\n\n\t\t3.Check Availability\n\n\t\t4.Issue a Book\n\n\t\t5.Calculate fine\n\n\t\t6.Clear fine amount\n\n\t\t7.Return book\n\n\t\t8.Logout\n\n\t\t9.Close Application\n";
    cout << "\n\t\tEnter your choice : ";
    cin >> i;
    system("cls");
    issuedbooks(this);
    switch (i)
    {
    case 1:
    {
        booklist();
        string ans;
        cout << "\n\t\tWould you like to issue any book? (Y/N) : ";
        cin >> ans;
        if (ans == "Y")
        {
            issue(this);
        }
        else
        {
            exit(0);
        }
        break;
    }
    case 2:
    {
        int c;
        for (int k = 0; k < (this->books).size(); k++)
        {
            show(&((this->books)[k]));
        }
        cout << "\n\t\t>>Please Choose One Option:\n";
        cout << "\n\t\t1.Show Duedate\n\n\t\t2.Return Book\n\n\t\t3.Go back\n";
        cout << "\n\t\tEnter your choice : ";
        cin >> c;
        if (c == 2)
        {
            rtrn(this);
        }
        else if (c == 1)
        {
            vector<string> v;
            int k;
            stringstream ss(search());
            while (ss.good())
            {
                string substr;
                getline(ss, substr, ',');
                v.push_back(substr);
            }
            k = stoi(v[0]);
            cout << "\n\t\tDue_date : " << bookv[k].Duedate << endl;
        }
        else
        {
            prof();
        }
        break;
    }
    case 3:
    {
        vector<string> v;
        int k;
        string str1;
        stringstream ss(search());
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ',');
            v.push_back(substr);
        }
        k = stoi(v[0]);
        str1 = v[2];
        if (bookv[k].Available == "1")
        {
            cout << "\n\t\tBook is available!" << endl;
        }
        else
        {
            cout << "\n\t\tBook is not available!!" << endl;
        }
        break;
    }
    case 4:
    {
        issue(this);
        issuedbooks(this);
        break;
    }
    case 5:
    {
        int fine = 0;
        string str1;
        cout << "\n\t\tEnter today's date (dd/mm/yy) : ";
        cin >> str1;
        for (int k = 0; k < (this->books).size(); k++)
        {
            fine = fine + calc_fine(((this->books)[k]).Duedate, str1, 5);
        }
        this->Fine_amount = fine;
        cout << "\n\t\tYour total fine is : " << fine << endl;
        break;
    }
    case 6:
    {
        clear_fine(this);
        break;
    }
    case 7:
    {
        rtrn(this);
        break;
    }
    case 8:
    {
        logout();
        break;
    }
    case 9:
        exit(0);
        break;

    default:
        cout << "\n\t\tPlease enter correct option :(";
        cout << "\n\t\tPress any key to go back.....";
        getch();
        system("cls");
        prof();
    }
}

void Student::stud()
{
    int i;
    cout << "\n\t************ WELCOME ************\n";
    cout << "\n\t\t>>Please Choose One Option:\n";
    cout << "\n\t\t1.View BookList\n\n\t\t2.View Your Issued Books\n\n\t\t3.Check Availability\n\n\t\t4.Issue a Book\n\n\t\t5.Calculate fine\n\n\t\t6.Clear fine amount\n\n\t\t7.Return book\n\n\t\t8.Logout\n\n\t\t9.Close Application\n";
    cout << "\n\t\tEnter your choice : ";
    cin >> i;
    system("cls");
    issuedbooks(this);
    switch (i)
    {
    case 1:
        booklist();
        break;
    case 2:
    {
        int c;
        for (int k = 0; k < (this->books).size(); k++)
        {
            show(&((this->books)[k]));
        }
        cout << "\n\t\t>>Please Choose One Option:\n";
        cout << "\n\t\t1.Show Duedate\n\n\t\t2.Return Book\n\n\t\t3.Go back\n";
        cout << "\n\t\tEnter your choice : ";
        cin >> c;
        if (c == 2)
        {
            rtrn(this);
        }
        else if (c == 1)
        {
            vector<string> v;
            int k;
            stringstream ss(search());
            while (ss.good())
            {
                string substr;
                getline(ss, substr, ',');
                v.push_back(substr);
            }
            k = stoi(v[0]);
            cout << "\n\t\tDue_date : " << bookv[k].Duedate << endl;
        }
        else
        {
            stud();
        }
        break;
    }
    case 3:
    {
        system("cls");
        vector<string> v;
        int k;
        string str1;
        stringstream ss(search());
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ',');
            v.push_back(substr);
        }
        k = stoi(v[0]);
        str1 = v[2];
        if (bookv[k].Available == "1")
        {
            cout << "\n\t\tBook is available!" << endl;
        }
        else
        {
            cout << "\n\t\tBook is not available!!" << endl;
        }
        break;
    }
    case 4:
    {
        if ((this->books).size() < 5)
        {
            issue(this);
            issuedbooks(this);
        }
        else
        {
            cout << "\n\t\tYou have reached maximum limit and hence cannot issue any book !" << endl;
        }
        break;
    }
    case 5:
    {
        int fine = 0;
        string str1;
        cout << "\n\t\tEnter today's date (dd/mm/yy) : ";
        cin >> str1;
        for (int k = 0; k < (this->books).size(); k++)
        {
            fine = fine + calc_fine(((this->books)[k]).Duedate, str1, 2);
        }
        this->Fine_amount = fine;
        cout << "\n\t\tYour total fine is : " << fine << endl;
        break;
    }
    case 6:
    {
        clear_fine(this);
        break;
    }
    case 7:
        rtrn(this);
        break;
    case 8:
        logout();
        break;
    case 9:
        exit(0);
    default:
        cout << "\n\t\tPlease enter correct option :(";
        cout << "\n\t\tPress any key to go back.....";
        getch();
        system("cls");
        stud();
    }
}

void Librarian ::lib()
{
    system("cls");
    int i;
    cout << "\n\t************ WELCOME ************\n";
    cout << "\n\t\t>>Please Choose One Option:\n";
    cout << "\n\t\t1.View BookList\n\n\t\t2.View Users\n\n\t\t3.Add/Update/Delete/Search Book\n\n\t\t4.Add/Update/Delete/Search User\n\n\t\t5.View Issued Books\n\n\t\t6.See who have issued books\n\n\t\t7.Logout\n\n\t\t8.Close Application\n";
    cout << "\n\t\tEnter your choice : ";
    cin >> i;
    system("cls");
    switch (i)
    {
    case 1:
    {
        booklist();
        string ans;
        cout << "\n\t\tWould you like to add/update/delete/search any book? (Y/N) : ";
        cin >> ans;
        if (ans == "Y")
        {
            modify();
        }
        else
        {
            lib();
        }
        break;
    }
    case 2:
    {
        userlist();
        string ans;
        cout << "\n\t\tWould you like to add/update/delete/search any user? (Y/N) : ";
        cin >> ans;
        if (ans == "Y")
        {
            modifyuser();
        }
        else
        {
            lib();
        }
        break;
    }
    case 3:
        modify();
        break;
    case 4:
        modifyuser();
        break;
    case 5:
    {
        int flag = 0;

        for (int k = 0; k < bookv.size(); k++)
        {
            if (bookv[k].Available == "0")
            {
                if (flag == 0)
                {
                    cout << "\n\t\tHere is the list of issued books:- " << endl;
                    flag = 1;
                }
                show(&(bookv[k]));
            }
        }
        if (flag == 0)
        {
            cout << "\n\t\tNo books issued yet !!\n"
                 << endl;
        }
        break;
    }
    case 6:
    {
        int k, comp;
        vector<string> v;
        string str1, str2;
        stringstream ss(search());
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ',');
            v.push_back(substr);
        }
        k = stoi(v[0]);
        str1 = v[2];
        comp = v[1] == "1" ? 0 : 2;
        if (k != bookv.size())
        {
            cout << "\n\t\tBook has been issued by :-\n";
            cout << "\n\t\tUsername : " << bookv[k].Borrower << "\n";
            cout << "\n\t\tID : " << bookv[k].ID << "\n";
        }
        else
        {
            cout << "\n\t\tBook has not been issued.\n";
        }
        break;
    }
    case 7:
    {
        system("cls");
        logout();
    }
    case 8:
        exit(0);
    default:
        cout << "\n\t\tPlease enter correct option :(";
        cout << "\n\t\tPress any key to go back.....";
        getch();
        system("cls");
        lib();
    }
}

void logout()
{
    Select_role();
}

void login(string selrole)
{
    int i;
    system("cls");
    Professor obj1;
    Student obj2;
    Librarian obj3;
    string pwdreal;
    cout << "\n\t\tEnter Username : ";
    string uname;
    cin >> uname;

    for (i = 0; i < vecu.size(); i++)
    {
        if (vecu[i][0] == uname)
        {
            if (vecu[i][3] == selrole)
            {
                if (selrole == "1")
                {
                    obj1.username = uname;
                    obj1.ID = vecu[i][1];
                    obj1.password = vecu[i][2];
                    obj1.role = vecu[i][3];
                    pwdreal = vecu[i][2];
                }
                else if (selrole == "2")
                {
                    obj2.username = uname;
                    obj2.ID = vecu[i][1];
                    obj2.password = vecu[i][2];
                    obj2.role = vecu[i][3];
                    pwdreal = vecu[i][2];
                }
                else
                {
                    obj3.username = uname;
                    obj3.ID = vecu[i][1];
                    obj3.password = vecu[i][2];
                    obj3.role = vecu[i][3];
                    pwdreal = vecu[i][2];
                }
                vecu = openuser();
                break;
            }
            else
            {
                cout << "\n\t\tInvalid User";
                cout << "\n\t\tPress any key to go back.....";
                getch();
                system("cls");
                login(selrole);
            }
        }
    }
    if (i == vecu.size())
    {
        cout << "\n\t\tUser does not exist !" << endl;
        cout << "\n\t\tPress any key to go back.....";
        getch();
        system("cls");
        Select_role();
    }
    else
    {
        cout << "\n\t\tEnter Password : ";
        string pwd;
        cin >> pwd;
        if (pwd == pwdreal)
        {
            system("cls");
            if (selrole == "3")
            {

                obj3.lib();
            }
            else if (selrole == "2")
            {
                obj2.stud();
            }
            else
            {
                obj1.prof();
            }
        }
        else
        {
            cout << "\n\n\t\tInvalid userame or password\n\n\t\tTry Again";
            cout << "\n\t\tPress any key to go back.....";
            getch();
            system("cls");
            Select_role();
        }
    }
}

void Select_role()
{
    int i;
    cout << "\n\t\t*********** LIBRARY MANAGEMENT SYSTEM ***********\n";
    cout << "\n\t\t>>Please Choose Any Option To login \n";
    cout << "\n\t\t1.Professor\n\n\t\t2.Student\n\n\t\t3.Librarian\n\n\t\t4.Close Application\n";
    cout << "\n\t\tEnter your choice : ";
    cin >> i;
    if (i == 1 || i == 2 || i == 3)
    {
        system("cls");
        login(to_string(i));
    }
    else if (i == 4)
        exit(0);
    else
    {
        cout << "\n\t\tPlease enter correct option :(";
        cout << "\n\t\tPress any key to go back.....";
        getch();
        system("cls");
        Select_role();
    }
}

int main()
{
    vecb = openbook();
    vecu = openuser();
    uservec();
    bookvec();
    Select_role();
}
