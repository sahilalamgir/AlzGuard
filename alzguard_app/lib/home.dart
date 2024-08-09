import 'package:flutter/material.dart';

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        resizeToAvoidBottomInset: true,
        backgroundColor: Colors.blue[400],
        appBar: AppBar(
          backgroundColor: Colors.white,
          title: Center(
            child: Text('AlzGuard',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: const Color.fromARGB(255, 3, 63, 105),
                  fontSize: 22,
                )),
          ),
          elevation: 20,
        ),
        body: SingleChildScrollView(
          child: Center(
              child: Column(
            children: [
              Padding(
                  padding: const EdgeInsets.all(15.0),
                  child: ElevatedButton(
                      onPressed: () {},
                      child: Text('Input MRI Scan',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          )))),
              Image.asset(
                'assets/img/4751.png',
                scale: 0.6,
              ),
              SizedBox(
                height: 10,
              ),
              Text(
                'Input Clinical Data Below:',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 18,
                  color: const Color.fromARGB(255, 22, 94, 152),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(25, 15, 25, 0),
                child: TextField(
                  controller: TextEditingController(),
                  decoration: const InputDecoration(
                      hintText: 'Input Age..',
                      contentPadding: EdgeInsets.fromLTRB(10, 1, 2, 0),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.all(Radius.circular(4.0)),
                      )),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(25, 15, 25, 0),
                child: TextField(
                  controller: TextEditingController(),
                  decoration: const InputDecoration(
                      hintText: 'Input Cholesterol HDL..',
                      contentPadding: EdgeInsets.fromLTRB(10, 1, 2, 0),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.all(Radius.circular(4.0)),
                      )),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(25, 15, 25, 0),
                child: TextField(
                  controller: TextEditingController(),
                  decoration: const InputDecoration(
                      hintText: 'Input MMSE..',
                      contentPadding: EdgeInsets.fromLTRB(10, 1, 2, 0),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.all(Radius.circular(4.0)),
                      )),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(25, 15, 25, 0),
                child: TextField(
                  controller: TextEditingController(),
                  decoration: const InputDecoration(
                      hintText: 'Input Functional Assessment..',
                      contentPadding: EdgeInsets.fromLTRB(10, 1, 2, 0),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.all(Radius.circular(4.0)),
                      )),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(25, 15, 25, 0),
                child: TextField(
                  controller: TextEditingController(),
                  decoration: const InputDecoration(
                      hintText: 'Input Memory Complaints..',
                      contentPadding: EdgeInsets.fromLTRB(10, 1, 2, 0),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.all(Radius.circular(4.0)),
                      )),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(25, 15, 25, 0),
                child: TextField(
                  controller: TextEditingController(),
                  decoration: const InputDecoration(
                      hintText: 'Input Memory Complaints..',
                      contentPadding: EdgeInsets.fromLTRB(10, 1, 2, 0),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.all(Radius.circular(4.0)),
                      )),
                ),
              ),
              Padding(
                  padding: const EdgeInsets.all(15.0),
                  child: ElevatedButton(
                    onPressed: () {},
                    child: Text('Calculate!',
                        style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                            color: Colors.white)),
                    style: ElevatedButton.styleFrom(
                        backgroundColor:
                            const Color.fromARGB(255, 108, 158, 183),
                        shape: RoundedRectangleBorder(
                            borderRadius:
                                BorderRadius.all(Radius.circular(4)))),
                  )),
            ],
          )),
        ));
  }
}
