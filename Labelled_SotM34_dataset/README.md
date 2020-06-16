
# Labelled-SotM34-dataset

The selected experimental dataset for evaluating the UAHL is the <a href="http://honeynet.onofri.org/scans/scan34/" target="_blank">`SotM34`</a> dataset, which consists of heterogeneous log-files collected from a honeynet that was targeted by several attacks in which some of them have successfully managed to compromise the targeted device. The dataset includes Apache logs, Linux Syslog, Snort NIDS logs, and IPtables Firewall logs. Fig. 4 shows the time disruption of those entries.

<p align="center">
  <img width="550" src="../Docs/imgs/events_dist.png"><br/>  
  <sub>Figure 1 - SotM34 Dataset - Events’ Time Distribution</sub>
</p>

The dataset is provided from the source in its original format (TXT) without preprocessing or labelling for events. To the best of our knowledge, there is no available preprocessed version of the SotM34 dataset. Therefore, we have produced a preprocessed and labelled version of the dataset as illustrated in the following subsections.

---

## Dataset pre-processing:
  - Split log-files were combined into their main files.
  - The HTTP and SYSLOG events are differed by 287 minutes against the IPTables Firewall log. Thus, all times in the HTTP and SYSLOG files were modified (287 minutes added to all events).
  - Datetimes format was unified for all events in the dataset (i.e. ”%Y-%m-%d %H:%M:%S”)
  - Feature sets were assigned for the log-files based on their structure. The following table shows the lists of extracted features.

	<small>

	 | Log-file Name	| Set of Features |
	 |:----------------:|:---------------:|
	 |	HTTP_Access		|	Date~, Time~, ClientIP$, HTTP_method, ClientRequestLine@,<br> Http_protocol@, StatusCode, ObjectSize~, Referrer~, Agent@ |
	 |	HTTP_Error		|	Date~, Time~, Type, ClientIP$, Reason_Phrase, Message@ |
	 |	HTTP_SSL_Error	|	Date~, Time~, Type, Message@ |
	 |	SYSLOG_Messages	|	Date~, Time~, Logging_device~, Logging_Daemon, PID~,<br> Operation@, User, Tty, UID, EUID, Remotehost@, System_message@ |
	 |	SYSLOG_Secure	|	Date~, Time~, Logging_device~, Logging_Daemon,<br> PID~, Operation, User, Source$, Port~ |
	 |	SYSLOG_Mail		|	Date~, Time~, Logging_device~, Logging_Daemon, PID~, QID~, From,<br> To, Size~, Class, nrcpts, Protocol, Daemon, msgid~, relay, Ruleset,<br> arg1, Ctladdr@, delay@, xdelay@, mailer, pri~, reject@, dsn@, stat@ |
	 |	SNORT			|	Date~, Time~, Logging_device~, RuleNumber@, Rule@,<br>Classification~, Priority~, Protocol, SrcIP\$, SrcPort~, DstIP$, DstPort~ |
	</small>

  - Blank cells were replaced with hyphens ”-”  
  - Log-files were converted into CSV files. Thus they can be easily imported into our framework as dataframes. The sets of features are extracted based on documentations of the HTTP server, Snort, and Unix logging system. Furthermore, indicators were added at the end of feature names to identify types of data inside the columns (”@” for text data, ”~” for ordinal categorical data, ”$” for IP addresses, and none for nominal categorical data).

---

## Events Labelling:
Technical description reports <a href="../Analysis_reports/" target="_blank">`[Click here]`</a> have described, in detail, attacks and structure of the SotM34 dataset. The reports provide sources and types of attacks in the log-files, which were analysed by domain experts. These reports were cross validated and then used to label the dataset’s events. Consequently, 80591 events were labelled as shown in the following table:

| Log-file Name		| No. of Events | No. of Unique Labels 	|
|:-----------------:|:-------------:|:---------------------:|
|	HTTP_Access		|	3554		|			12			|
|	HTTP_Error		|	3692		|			12			|
|	HTTP_SSL_Error	|	374			|			4			|
|	SYSLOG_Messages	|	1166		|			11			|
|	SYSLOG_Secure	|	1587		|			6			|
|	SYSLOG_Mail		|	1172		|			7			|
|	SNORT			|	69039		|			26			|
|	**Total**		|	**80591**	|	 	  **78**		|

---

## Details of the labels used for each log-file:

- **HTTP_Access file:**
	-	AA: Failed access attempt to Files/Paths (status code: 404)
	-	AB: Failed access attempt to Files/Paths (status code: 400)
	-	AC: Failed "POST" requests (status code 404)
  	-	AD: CONNECT requests (status code 405)
	-	AE: Forbidden access attempt to the root directory
	-	AF: Forbidden access attempt to websites
	-	AG: Forbidden access attempt to other paths/files
	-	AH: Proxy request (status code: 404)
	-	AI: OPTIONS Scan (status code 200)
	-	AJ: Scan for vulnerable PHP scripts (status code: 404)
	-	AK: Scan for vulnerable PHP scripts (status code: 200)
	-	AL: Scan for vulnerable PHP scripts (status code: 500)


- **HTTP_Error_Log file:**
	-	BA: File/directory does not exist
	-	BB: Directory index forbidden by rule
	-	BC: Script not found or unable to stat
	-	BD: attempt to invoke directory as script
	-	BE: Attempt to serve directory
	-	BF: Request without hostname
	-	BG: sh connection errors (No such file or directory/no job control)
	-	BH: The timeout specified has expired
	-	BI: Broken pipe
	-	BJ: Errors/Messages while connections
	-	BK: System messages (notice: Apache server messages)
	-	BL: System messages (warn: Child process still did not exit)


- **HTTP_SSL_Error_Log file:**
	-	CA: SSL handshake failed/interrupted
	-	CB: Spurious SSL handshake interrupt
	-	CC: SSL Library Error
	-	CD: System messages (includes RSA_server_certificate messages)


- **Syslog_Messages file:**
	-	DA: Message repeated
	-	DB: Failed SSH login
	-	DC: CUPS messages (CUPS is the Common UNIX Print System) 
	-	DD: Syslog messages (syslog & syslogd & syslogd 1.4.1)
	-	DE: HTTP messages (Apache Server)
	-	DF: rpc.statd messages
	-	DG: kernel messages (including all subprocesses activated during the Kernel processes)
	-	DH: Xinetd messages
	-	DI: Session opened/closed (su daemon)
	-	DJ: Session opened/closed (sshd daemon)
	-	DK: Session opened/closed (login daemon)


- **Syslog_Secure file:**
	-	EA: Accepted password
	-	EB: Illegal user
	-	EC: Failed password
	-	ED: Did not receive identification string
	-	EE: Scanned
	-	EF: Xinetd messages


- **Syslog_Maillog file:**
	-	FA: Sent mails
	-	FB: Message accepted for delivery
	-	FC: Connection related messages (Alert: Did not issue MAIL/EXPN/VRFY/ETRN during connection to MTA)
	-	FD: Connection related messages (rejecting/lost connections)
	-	FE: Connection related messages (pop3 service messages)
	-	FF: Connection related messages (Other server messages)
	-	FG: Repeated messages


- **Snortlog file:**
  - GA: Spade: Closed dest port used
  - GB: Spade: Source used odd dest port
  - GC: Potential Corporate Privacy Violation
  - GD: Web Application Attack
  - GE: Access to a potentially vulnerable web application
  - GF: An attempted login using a suspicious username was detected
  - GG: Potentially Bad Traffic
  - GH: A Network Trojan was detected
  - GI: Detection messages (spp_stream4)
  - GJ: Attempted Information Leak - Automated NMAP TCP scan
  - GK: Attempted Information Leak - NMAP ICMP PING
  - GL: Attempted Information Leak - WEB-MISC http directory traversal
  - GM: Attempted Information Leak - DOUBLE DECODING ATTACK
  - GN: Attempted Information Leak - DNS named version attempt
  - GO: Executable code was detected (SHELLCODE x86 NOOP)
  - GP: Executable code was detected (BARE BYTE UNICODE ENCODING)
  - GQ: Detection of a Network Scan
  - GR: Decode of an RPC Query (UDP connection)
  - GS: Decode of an RPC Query (TCP connection)
  - GT: Email server errors
  - GU: Attempted Administrator Privilege Gain (UDP connection)
  - GV: Attempted Administrator Privilege Gain (TCP connection)
  - GW: ICMP Pings messages
  - GX: Attempted Denial of Service
  - GY: MS-SQL Worm alerts
  - GZ: Snort_decoder warning messages