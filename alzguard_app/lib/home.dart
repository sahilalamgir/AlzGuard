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
      backgroundColor: Colors.blueGrey[200],
      appBar: AppBar(
        backgroundColor: const Color.fromARGB(255, 57, 139, 206),
        title: Center(
          child: Text('AlzGuard',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: Colors.white,
                fontSize: 22,
              )),
        ),
        elevation: 20,
      ),
      body: Center(child: Column(
        children: [
          
        ],
      ))
    );
  }
}
